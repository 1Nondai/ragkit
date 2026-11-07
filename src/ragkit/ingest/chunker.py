"""切片：把 Section 切成"可检索的语义单元"，并生成稳定标识。"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence
from dataclasses import dataclass, field

from ragkit.ingest.loader import LoadedDocument

_SEPARATORS = ["\n\n", "\n", "。", "！", "？", ". ", "! ", "? ", "；", "; ", "，", ", ", " "]

@dataclass
class Chunk:
    """最小检索单元。"""

    text: str
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def chunk_id(self) -> str:
        """本块的主键（内容指纹），幂等入库与 wiki 反查都靠它。"""
        return str(self.metadata.get("chunk_id", ""))

def _split_once(text: str, size: int, overlap: int) -> list[str]:
    """单层递归切分：找不到分隔符时按长度硬切（兜底）。"""
    if len(text) <= size:
        return [text]

    for sep in _SEPARATORS:
        if sep in text:
            parts = text.split(sep)
            keep = sep.strip()
            merged = [p + keep for p in parts[:-1]] + [parts[-1]] if keep else parts
            return [p for p in merged if p.strip()]
    step = max(1, size - overlap)
    return [text[i : i + size] for i in range(0, len(text), step)]

def recursive_split(text: str, size: int, overlap: int) -> list[str]:
    """递归切分：大块 → 小块，直到每块都 <= size。"""
    if len(text) <= size:
        return [text.strip()] if text.strip() else []

    pieces = _split_once(text, size, overlap)
    out: list[str] = []
    for piece in pieces:
        if len(piece) <= size:
            if piece.strip():
                out.append(piece.strip())
        else:
            out.extend(recursive_split(piece, size, overlap))
    return _merge_with_overlap(out, size, overlap)

def _merge_with_overlap(pieces: Sequence[str], size: int, overlap: int) -> list[str]:
    """相邻小块合并到接近 size，并按 overlap 回带上一块的尾部。"""
    if overlap <= 0:
        return list(pieces)

    merged: list[str] = []
    for piece in pieces:
        if merged and len(merged[-1]) + len(piece) <= size:
            merged[-1] = merged[-1] + " " + piece
        else:
            if merged:
                tail = merged[-1][-overlap:]
                piece = tail + piece if not piece.startswith(tail) else piece
            merged.append(piece)
    return merged

def _normalize_for_hash(text: str) -> str:
    """归一化：去空白差异，避免"多了个空格"就生成新 chunk。"""
    return re.sub(r"\s+", " ", (text or "").strip())

def make_chunk_id(text: str, title: str = "", filename: str = "") -> str:
    """内容指纹：chunk 的主键。"""
    raw = f"{filename}|{title}|{_normalize_for_hash(text)}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:32]

def chunk_sections(
    doc: LoadedDocument,
    *,
    chunk_size: int = 800,
    chunk_overlap: int = 120,
    min_chunk_chars: int = 30,
) -> list[Chunk]:
    """把文档的 Section 切成 Chunk（带标题路径与稳定 ID）。"""
    chunks: list[Chunk] = []
    for idx, section in enumerate(doc.sections):
        title = section.title
        pieces = recursive_split(section.text, chunk_size, chunk_overlap)
        is_image = section.metadata.get("modality") == "image"
        for order, piece in enumerate(pieces):
            text = piece.strip()
            if not is_image and len(text) < min_chunk_chars:
                continue
            chunks.append(
                Chunk(
                    text=text,
                    metadata={
                        **section.metadata,
                        "chunk_id": make_chunk_id(text, title, doc.filename),
                        "doc_id": hashlib.sha1(doc.source.encode("utf-8")).hexdigest()[:16],
                        "source": doc.source,
                        "filename": doc.filename,
                        "filetype": doc.filetype,
                        "title": title,
                        "category": section.metadata.get("category", "content"),
                        "section_index": idx,
                        "chunk_index": len(chunks),
                        "order": order,
                        "chars": len(text),
                        "modality": section.metadata.get("modality", "text"),
                        "image_path": section.metadata.get("image_path", ""),
                        "page": section.metadata.get("page", 0),
                    },
                )
            )
    return chunks

def chunk_documents(
    docs: Sequence[LoadedDocument], *, chunk_size: int = 800, chunk_overlap: int = 120,
    min_chunk_chars: int = 30,
) -> list[Chunk]:
    """批量切片 + 按 chunk_id 去重（幂等入库的关键一步）。"""
    seen: set[str] = set()
    out: list[Chunk] = []
    for doc in docs:
        for chunk in chunk_sections(
            doc, chunk_size=chunk_size, chunk_overlap=chunk_overlap,
            min_chunk_chars=min_chunk_chars,
        ):
            if chunk.chunk_id in seen:
                continue
            seen.add(chunk.chunk_id)
            chunk.metadata["chunk_index"] = len(out)
            out.append(chunk)
    return out
