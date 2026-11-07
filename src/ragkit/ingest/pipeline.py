"""入库流水线：目录 → 加载 → 切片 → 向量化 → 写入。"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from ragkit.embeddings import Embeddings, ImageEmbeddings
from ragkit.ingest.chunker import Chunk, chunk_documents
from ragkit.ingest.loader import LoadedDocument, load_directory
from ragkit.logging_setup import get_logger
from ragkit.retrieval.protocol import VectorStore

if TYPE_CHECKING:
    from ragkit.multimodal.pdf_loader import PdfLoader

log = get_logger("ingest.pipeline")

@dataclass
class IngestStats:
    """入库统计。"""
    files: int = 0
    sections: int = 0
    chunks: int = 0
    written: int = 0
    skipped_files: int = 0
    elapsed_ms: float = 0.0
    collection: str = ""
    embedded_dim: int = 0
    image_vectors: int = 0

    def to_dict(self) -> dict:
        """转成可 JSON 序列化的统计（CLI / API 直接输出）。"""
        return {
            "files": self.files,
            "sections": self.sections,
            "chunks": self.chunks,
            "written": self.written,
            "skipped_files": self.skipped_files,
            "elapsed_ms": round(self.elapsed_ms, 1),
            "collection": self.collection,
            "embedded_dim": self.embedded_dim,
            "image_vectors": self.image_vectors,
        }

class IngestPipeline:
    """入库流水线。"""

    def __init__(
        self,
        store: VectorStore,
        embeddings: Embeddings,
        *,
        chunk_size: int = 800,
        chunk_overlap: int = 120,
        min_chunk_chars: int = 30,
        batch_size: int = 32,
        kb: str | None = None,
        pdf_loader: PdfLoader | None = None,
        image_root: Path | None = None,
    ) -> None:
        self.store = store
        self.embeddings = embeddings
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_chars = min_chunk_chars
        self.batch_size = batch_size
        self.kb = kb
        self.pdf_loader = pdf_loader
        self.image_root = Path(image_root) if image_root else None

    def load(self, data_dir: Path) -> list[LoadedDocument]:
        """加载 md/txt；配了 pdf_loader 时一并把 PDF 转成文档对象。"""
        data_dir = Path(data_dir)
        docs = load_directory(data_dir)
        if self.pdf_loader is not None:
            for pdf in sorted(data_dir.rglob("*.pdf")):
                try:
                    docs.append(self.pdf_loader.load(pdf))
                except Exception as exc:  # noqa: BLE001 —— 单文件失败只跳过
                    log.warning("PDF 解析失败，跳过 %s: %s", pdf.name, exc)
        log.info("加载完成: %d 个文件, %d 个段落", len(docs), sum(d.section_count for d in docs))
        return docs

    def chunk(self, docs: Sequence[LoadedDocument]) -> list[Chunk]:
        """切块（顺带把 kb 标签写进 section.metadata，随 chunk 透传）。"""
        if self.kb:
            for doc in docs:
                for section in doc.sections:
                    section.metadata["kb"] = self.kb
        chunks = chunk_documents(
            docs,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            min_chunk_chars=self.min_chunk_chars,
        )
        log.info("切片完成: %d 个 chunk（平均 %.0f 字）",
                 len(chunks), sum(len(c.text) for c in chunks) / (len(chunks) or 1))
        return chunks

    def embed(self, chunks: Sequence[Chunk]) -> list[list[float]]:
        """批量向量化（带进度日志）。"""
        texts = [c.text for c in chunks]
        vectors: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            vectors.extend(self.embeddings.embed_documents(texts[i : i + self.batch_size]))
            if (i // self.batch_size) % 5 == 0:
                log.info("向量化进度: %d/%d", len(vectors), len(texts))
        return vectors

    def _embed_images(self, chunks: Sequence[Chunk]) -> tuple[list[list[float] | None] | None, int]:
        """给图片块算图片向量（与 chunks 对齐，文本块位置为 None）。

        三个条件缺一就整体跳过：embedding 不支持图片 / 没配 image_root / 没有图片块。
        单个图读失败只告警，不影响其他块与整条入库。
        """
        if not isinstance(self.embeddings, ImageEmbeddings) or self.image_root is None:
            return None, 0
        out: list[list[float] | None] = []
        n = 0
        for c in chunks:
            vec: list[float] | None = None
            rel = c.metadata.get("image_path") or ""
            if rel:
                try:
                    data = (self.image_root / rel).read_bytes()
                    vec = self.embeddings.embed_image(data)
                    n += 1
                except Exception as exc:  # noqa: BLE001 —— 单个图失败不影响入库
                    log.warning("图片向量失败，跳过 %s: %s", rel, exc)
            out.append(vec)
        if n:
            log.info("图片向量: %d 个", n)
        return (out if n else None), n

    def write(
        self,
        chunks: Sequence[Chunk],
        vectors: Sequence[Sequence[float]],
        image_vectors: Sequence[Sequence[float] | None] | None = None,
    ) -> int:
        """分批写库（批大小受 embedding/写入两端限制）。"""
        written = 0
        for i in range(0, len(chunks), self.batch_size):
            written += self.store.upsert(
                list(chunks[i : i + self.batch_size]), list(vectors[i : i + self.batch_size]),
                list(image_vectors[i : i + self.batch_size]) if image_vectors is not None else None,
            )
        return written

    def run(self, data_dir: Path, *, recreate: bool = False) -> IngestStats:
        """跑完整流水线。"""
        started = time.perf_counter()
        data_dir = Path(data_dir)
        if not data_dir.exists():
            raise FileNotFoundError(f"数据目录不存在: {data_dir}")

        import os

        file_count = sum(len(files) for _, _, files in os.walk(data_dir))

        self.store.ensure_collection(dim=self.embeddings.dim, recreate=recreate)

        docs = self.load(data_dir)
        chunks = self.chunk(docs)
        vectors = self.embed(chunks)
        image_vectors, n_image = self._embed_images(chunks)
        written = self.write(chunks, vectors, image_vectors)

        stats = IngestStats(
            files=len(docs),
            sections=sum(d.section_count for d in docs),
            chunks=len(chunks),
            written=written,
            skipped_files=max(0, file_count - len(docs)),
            elapsed_ms=(time.perf_counter() - started) * 1000,
            collection=getattr(self.store, "collection_name", "memory"),
            embedded_dim=len(vectors[0]) if vectors else 0,
            image_vectors=n_image,
        )
        log.info("入库完成: %s", stats.to_dict())
        return stats
