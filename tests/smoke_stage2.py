"""阶段 2 全链路冒烟：图片块走完整图（route→wiki→rewrite→retrieve→grade→generate→verify）。"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ragkit.embeddings import HashingEmbedding
from ragkit.graph.graph import build_graph, run
from ragkit.graph.nodes import Deps
from ragkit.ingest.chunker import Chunk
from ragkit.llm import ExtractiveLLM
from ragkit.retrieval.protocol import InMemoryStore
from ragkit.retrieval.service import RetrievalService

PNG = b"\x89PNG\r\n\x1a\n" + b"q" * 200
tmp = Path(tempfile.mkdtemp(prefix="stage2_smoke_"))
(tmp / "images").mkdir()
(tmp / "images" / "3-2.png").write_bytes(PNG)

emb = HashingEmbedding(dim=128)
store = InMemoryStore()
store.ensure_collection(dim=128)
chunks = [
    Chunk(text="第三章讲有界流处理与水位线机制，正文内容足够长，可以正常检索命中。",
          metadata={"chunk_id": "c1", "filename": "ch3.pdf", "title": "3.1 水位线",
                    "source": "ch3.pdf", "category": "content",
                    "modality": "text", "image_path": "", "page": 5}),
    Chunk(text="图中展示了 JobManager 与 TaskManager 之间的 RPC 交互时序。",
          metadata={"chunk_id": "c2", "filename": "ch3.pdf", "title": "图 3-2 (P12)",
                    "source": "ch3.pdf", "category": "figure",
                    "modality": "image", "image_path": "images/3-2.png", "page": 12}),
]
store.upsert(chunks, emb.embed_documents([c.text for c in chunks]))
service = RetrievalService(store=store, embeddings=emb, top_k=3)

class VisionChat:
    name = "vision"
    seen_images = 0

    def invoke(self, prompt: str) -> str:
        return "纯文本答案"

    def invoke_messages(self, messages: list[dict]) -> str:
        for m in messages:
            if isinstance(m.get("content"), list):
                VisionChat.seen_images += sum(
                    1 for b in m["content"] if b.get("type") == "image_url"
                )
        return "视觉答案：JobManager 与 TaskManager 通过 RPC 通信。"

deps = Deps(service=service, chat=VisionChat(), image_root=tmp)  # type: ignore[arg-type]
state = run("JobManager 和 TaskManager 怎么通信", deps, session_id="smoke",
            compiled=build_graph(deps))
print("trace:", state.get("trace"))
print("answer:", state.get("answer"))
print("citations:", [(c.get("source"), c.get("title"), c.get("image_path"))
                    for c in state.get("citations", [])])
print("视觉图传入次数:", VisionChat.seen_images)
print("hits:", [(h.get("modality"), h.get("image_path")) for h in state.get("hits", [])])

deps2 = Deps(service=service, chat=ExtractiveLLM(), image_root=tmp)  # type: ignore[arg-type]
state2 = run("JobManager 和 TaskManager 怎么通信", deps2, session_id="smoke2",
             compiled=build_graph(deps2))
print("\n离线模式 trace:", state2.get("trace"))
print("离线模式 answer 前 80 字:", (state2.get("answer") or "")[:80])
