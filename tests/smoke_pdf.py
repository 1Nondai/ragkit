"""真实数据冒烟：Flink PDF + dots.ocr 预解析 JSON，离线跑通 解析→切片→入库→检索。

用法：.venv/Scripts/python.exe smoke_pdf.py
"""
from __future__ import annotations

import json
import sys
import tempfile
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ragkit.embeddings import HashingEmbedding
from ragkit.ingest.chunker import chunk_documents
from ragkit.multimodal.image_store import ImageStore
from ragkit.multimodal.ocr_client import OcrBlock
from ragkit.multimodal.pdf_loader import PdfLoader
from ragkit.retrieval.protocol import InMemoryStore

ZIP = Path(r"C:/Users/DDY18/Desktop/10月18日后-多模态RAG项目补充代码.zip")
PREFIX = "Multimodal_RAG/dots_ocr/"
PDF_IN_ZIP = PREFIX + "第二章 Apache Flink快速入门.pdf"
OUT_DIR_PREFIX = PREFIX + "output/第二章 Apache Flink快速入门/"

scratch = Path(tempfile.mkdtemp(prefix="mmrag_smoke_"))
print("工作目录:", scratch)

with zipfile.ZipFile(ZIP) as z:
    names = z.namelist()
    pdf_pages = sorted(
        n for n in names
        if n.startswith(OUT_DIR_PREFIX) and n.endswith(".json") and "_page_" in n
    )
    print(f"zip 里预解析页数: {len(pdf_pages)}")
    pdf_path = scratch / "book.pdf"
    pdf_path.write_bytes(z.read(PDF_IN_ZIP))
    blocks_cache: dict[int, list[OcrBlock]] = {}
    bad_pages: list[tuple[int, str]] = []
    for n in pdf_pages:
        page_no = int(n.split("_page_")[1].split(".")[0]) + 1
        try:
            data = json.loads(z.read(n).decode("utf-8"))
            if isinstance(data, str):
                data = json.loads(data)
            if not isinstance(data, list):
                raise ValueError("顶层不是列表")
            blocks_cache[page_no] = [
                OcrBlock(category=b.get("category", "Text"), text=b.get("text", ""),
                         bbox=tuple(b["bbox"]) if b.get("bbox") else None)
                for b in data if isinstance(b, dict)
            ]
        except Exception as exc:  # noqa: BLE001 —— dots 输出烂 JSON 是已知现象
            bad_pages.append((page_no, str(exc)[:40]))
    print(f"可用页 {len(blocks_cache)}，坏页 {len(bad_pages)}: {bad_pages}")

store = ImageStore(root=scratch / "data")
loader = PdfLoader(ocr=None, image_store=store, captioner=None)
doc = loader.load(pdf_path, blocks_cache=blocks_cache)
mod_counts = {}
for s in doc.sections:
    mod_counts[s.metadata.get("modality", "text")] = mod_counts.get(s.metadata.get("modality", "text"), 0) + 1
print(f"\n[路径A 版面解析] sections={len(doc.sections)} 按 modality: {mod_counts}")
img_sections = [s for s in doc.sections if s.metadata.get("modality") == "image"]
print(f"图片块 {len(img_sections)} 个，前 3 个 title:",
      [s.title_path[-1][:20] if s.title_path else "(无)" for s in img_sections[:3]])

chunks = chunk_documents([doc], chunk_size=800, chunk_overlap=120, min_chunk_chars=30)
c_mod = {}
for c in chunks:
    c_mod[c.metadata.get("modality", "text")] = c_mod.get(c.metadata.get("modality", "text"), 0) + 1
print(f"切片后 chunks={len(chunks)} 按 modality: {c_mod}")

emb = HashingEmbedding(dim=128)
store_v = InMemoryStore()
store_v.ensure_collection(dim=128)
store_v.upsert(chunks, emb.embed_documents([c.text for c in chunks]))

for q in ["Flink 是什么", "WordCount 程序怎么写", "JobManager 和 TaskManager"]:
    hits = store_v.search(q, emb.embed_query(q), top_k=3)
    hit_mod = [h.modality for h in hits]
    print(f"\n查询「{q}」top3 modality: {hit_mod}")
    for h in hits[:2]:
        print("   ", h.title[:24] or "(无标题)", "|", h.text[:44].replace("\n", " "))

doc_b = PdfLoader(ocr=None).load(pdf_path)
print(f"\n[路径B 文本层兜底] sections={len(doc_b.sections)}, 首 section 标题: {doc_b.sections[0].title[:24] if doc_b.sections and doc_b.sections[0].title else '(空)'}")
print("文件数检查: data/images 下文件数 =", len(list((scratch / "data" / "images").glob("*.png"))) if (scratch / "data" / "images").exists() else 0)
