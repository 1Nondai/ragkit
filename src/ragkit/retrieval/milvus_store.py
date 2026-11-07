"""Milvus 实现：稠密 + BM25 稀疏混合检索（生产路径）。"""

from __future__ import annotations

import logging
from collections.abc import Sequence

from ragkit.ingest.chunker import Chunk
from ragkit.retrieval.protocol import SearchHit, _rrf_fuse

logger = logging.getLogger("ragkit.retrieval.milvus")

FIELD_ID = "id"
FIELD_TEXT = "text"
FIELD_DENSE = "dense"
FIELD_SPARSE = "sparse"
FIELD_MODALITY = "modality"
FIELD_IMAGE_PATH = "image_path"
FIELD_PAGE = "page"
FIELD_DENSE_IMG = "dense_img"

class MilvusStore:
    """基于 pymilvus 的混合检索存储。"""

    def __init__(
        self,
        uri: str,
        collection: str,
        *,
        user: str | None = None,
        password: str | None = None,
        database: str = "default",
        timeout: float = 20.0,
        metric: str = "IP",
        consistency_level: str = "Bounded",
    ) -> None:
        self.uri = uri
        self.collection_name = collection
        self.user = user
        self.password = password
        self.database = database
        self.timeout = timeout
        self.metric = metric
        self.consistency_level = consistency_level
        self._client = None

    def client(self):
        """惰性建连：构造对象不产生网络调用（便于测试与配置校验）。"""
        if self._client is None:
            from pymilvus import MilvusClient

            kwargs = {"uri": self.uri, "timeout": self.timeout}
            if self.user:
                kwargs["user"] = self.user
            if self.password:
                kwargs["password"] = self.password
            if self.database:
                kwargs["db_name"] = self.database
            self._client = MilvusClient(**kwargs)
        return self._client

    def has_collection(self) -> bool:
        """集合是否已存在（建库/重建前判断）。"""
        return self.collection_name in self.client().list_collections()

    def ensure_collection(self, dim: int, *, recreate: bool = False) -> None:
        """建集合（幂等）。recreate=True 时先删除重建。"""
        from pymilvus import DataType, Function, FunctionType

        client = self.client()
        if recreate and self.has_collection():
            client.drop_collection(collection_name=self.collection_name)

        if self.has_collection():
            return

        schema = client.create_schema(auto_id=True, enable_dynamic_field=True)
        schema.add_field(FIELD_ID, DataType.INT64, is_primary=True, auto_id=True)
        schema.add_field(
            FIELD_TEXT, DataType.VARCHAR, max_length=8000,
            enable_analyzer=True, analyzer_params={"tokenizer": "jieba"},
        )
        schema.add_field("chunk_id", DataType.VARCHAR, max_length=64)
        schema.add_field("doc_id", DataType.VARCHAR, max_length=32)
        schema.add_field("filename", DataType.VARCHAR, max_length=512)
        schema.add_field("title", DataType.VARCHAR, max_length=1024)
        schema.add_field("category", DataType.VARCHAR, max_length=64)
        schema.add_field("source", DataType.VARCHAR, max_length=1024)
        schema.add_field(FIELD_MODALITY, DataType.VARCHAR, max_length=16)
        schema.add_field(FIELD_IMAGE_PATH, DataType.VARCHAR, max_length=512)
        schema.add_field(FIELD_PAGE, DataType.INT32)
        schema.add_field(FIELD_SPARSE, DataType.SPARSE_FLOAT_VECTOR)
        schema.add_field(FIELD_DENSE, DataType.FLOAT_VECTOR, dim=dim)
        schema.add_field(FIELD_DENSE_IMG, DataType.FLOAT_VECTOR, dim=dim)

        schema.add_function(
            Function(
                name="text_bm25_emb",
                input_field_names=[FIELD_TEXT],
                output_field_names=[FIELD_SPARSE],
                function_type=FunctionType.BM25,
            )
        )

        index_params = client.prepare_index_params()
        index_params.add_index(
            field_name=FIELD_SPARSE,
            index_type="SPARSE_INVERTED_INDEX",
            metric_type="BM25",
            params={"inverted_index_algo": "DAAT_MAXSCORE", "bm25_k1": 1.2, "bm25_b": 0.75},
        )
        index_params.add_index(
            field_name=FIELD_DENSE,
            index_type="HNSW",
            metric_type=self.metric,
            params={"M": 16, "efConstruction": 200},
        )
        index_params.add_index(
            field_name=FIELD_DENSE_IMG,
            index_type="HNSW",
            metric_type=self.metric,
            params={"M": 16, "efConstruction": 200},
        )
        client.create_collection(
            collection_name=self.collection_name, schema=schema, index_params=index_params
        )

    def upsert(
        self,
        chunks: Sequence[Chunk],
        vectors: Sequence[Sequence[float]],
        image_vectors: Sequence[Sequence[float]] | None = None,
    ) -> int:
        """批量写入。chunk_id 相同的新数据会覆盖旧的（幂等入库的基础）。

        image_vectors 与 chunks 对齐（文本块位置给 None）；Milvus 向量字段
        不可空，没有图向量的块写零向量 —— 图片检索时靠 modality 过滤把它们排除。
        """
        if len(chunks) != len(vectors):
            raise ValueError("chunks 与 vectors 长度不一致")
        if image_vectors is not None and len(image_vectors) != len(chunks):
            raise ValueError("image_vectors 与 chunks 长度不一致")
        if not chunks:
            return 0

        rows = []
        for i, (chunk, vec) in enumerate(zip(chunks, vectors)):
            meta = chunk.metadata
            img_vec = (image_vectors[i] if image_vectors is not None else None)
            rows.append(
                {
                    FIELD_TEXT: chunk.text,
                    FIELD_DENSE: list(vec),
                    FIELD_DENSE_IMG: list(img_vec) if img_vec else [0.0] * len(vec),
                    "chunk_id": meta.get("chunk_id", ""),
                    "doc_id": meta.get("doc_id", ""),
                    "filename": str(meta.get("filename", "")),
                    "title": str(meta.get("title", ""))[:1000],
                    "category": str(meta.get("category", "content")),
                    "source": str(meta.get("source", ""))[:1000],
                    FIELD_MODALITY: str(meta.get("modality", "text")),
                    FIELD_IMAGE_PATH: str(meta.get("image_path", ""))[:500],
                    FIELD_PAGE: int(meta.get("page", 0) or 0),
                    **({"kb": str(meta["kb"])} if meta.get("kb") else {}),
                }
            )
        self.client().insert(collection_name=self.collection_name, data=rows)
        return len(rows)

    def search(
        self,
        query_text: str,
        query_vector: Sequence[float],
        top_k: int = 5,
        *,
        filters: dict[str, object] | None = None,
        rrf_k: int = 60,
        dense_weight: float = 0.6,
        sparse_weight: float = 0.4,
    ) -> list[SearchHit]:
        """两路召回 + RRF 融合。"""
        client = self.client()
        expr = self._build_expr(filters)
        candidate_k = max(top_k * 4, 20)
        output_fields = [
            "chunk_id", "text", "filename", "title", "category", "source", "doc_id",
            FIELD_MODALITY, FIELD_IMAGE_PATH, FIELD_PAGE,
        ]

        dense_res = client.search(
            collection_name=self.collection_name,
            data=[list(query_vector)],
            anns_field=FIELD_DENSE,
            limit=candidate_k,
            search_params={"metric_type": self.metric, "params": {"ef": 64}},
            output_fields=output_fields,
            filter=expr,
            consistency_level=self.consistency_level,
        )
        sparse_res = client.search(
            collection_name=self.collection_name,
            data=[query_text],
            anns_field=FIELD_SPARSE,
            limit=candidate_k,
            search_params={"metric_type": "BM25"},
            output_fields=output_fields,
            filter=expr,
            consistency_level=self.consistency_level,
        )

        chunks_by_id: dict[str, Chunk] = {}
        dense_ranked: list[tuple[str, float]] = []
        sparse_ranked: list[tuple[str, float]] = []

        for hit in (dense_res[0] if dense_res else []):
            cid = hit["entity"].get("chunk_id")
            chunks_by_id[cid] = self._to_chunk(hit)
            dense_ranked.append((cid, float(hit["distance"])))
        for hit in (sparse_res[0] if sparse_res else []):
            cid = hit["entity"].get("chunk_id")
            chunks_by_id.setdefault(cid, self._to_chunk(hit))
            sparse_ranked.append((cid, float(hit["distance"])))

        return _rrf_fuse(chunks_by_id, dense_ranked, sparse_ranked, top_k=top_k, rrf_k=rrf_k)

    def search_by_image(
        self,
        query_vector: Sequence[float],
        top_k: int = 5,
        *,
        filters: dict[str, object] | None = None,
    ) -> list[SearchHit]:
        """图搜图：在 dense_img 上做单路 ANN。

        强制叠加 modality=="image" 过滤：文本块的 dense_img 是零向量，
        不过滤的话它们会以 0 分混进候选。用户自定义过滤条件与之合并。
        """
        merged = dict(filters or {})
        merged.setdefault(FIELD_MODALITY, "image")
        expr = self._build_expr(merged)
        client = self.client()
        candidate_k = max(top_k * 4, 20)
        res = client.search(
            collection_name=self.collection_name,
            data=[list(query_vector)],
            anns_field=FIELD_DENSE_IMG,
            limit=candidate_k,
            search_params={"metric_type": self.metric, "params": {"ef": 64}},
            output_fields=[
                "chunk_id", "text", "filename", "title", "category", "source", "doc_id",
                FIELD_MODALITY, FIELD_IMAGE_PATH, FIELD_PAGE,
            ],
            filter=expr,
            consistency_level=self.consistency_level,
        )
        hits: list[SearchHit] = []
        for raw in res[0] if res else []:
            chunk = self._to_chunk(raw)
            score = float(raw["distance"])
            hits.append(
                SearchHit(
                    chunk_id=chunk.metadata["chunk_id"],
                    text=chunk.text,
                    score=score,
                    metadata=chunk.metadata,
                    dense_score=score,
                    sparse_score=0.0,
                    rank_fusion="image",
                )
            )
        return hits[:top_k]

    @staticmethod
    def _to_chunk(hit) -> Chunk:
        entity = hit["entity"]
        meta = {
            "chunk_id": entity.get("chunk_id", ""),
            "doc_id": entity.get("doc_id", ""),
            "filename": entity.get("filename", ""),
            "title": entity.get("title", ""),
            "category": entity.get("category", "content"),
            "source": entity.get("source", ""),
            "modality": entity.get(FIELD_MODALITY, "text"),
            "image_path": entity.get(FIELD_IMAGE_PATH, ""),
            "page": int(entity.get(FIELD_PAGE, 0) or 0),
        }
        return Chunk(text=entity.get("text", ""), metadata=meta)

    @staticmethod
    def _build_expr(filters: dict[str, object] | None) -> str:
        """把 dict 过滤条件拼成 Milvus 表达式（只支持等值，够用且安全）。"""
        if not filters:
            return ""
        parts = []
        for key, value in filters.items():
            if isinstance(value, str):
                escaped = value.replace('"', '\\"')
                parts.append(f'{key} == "{escaped}"')
            else:
                parts.append(f"{key} == {value}")
        return " and ".join(parts)

    def count(self) -> int:
        """集合内实体数（/health 用）。"""
        if not self.has_collection():
            return 0
        res = self.client().query(
            collection_name=self.collection_name, filter="", output_fields=["count(*)"]
        )
        return int(res[0]["count(*)"]) if res else 0

    def drop(self) -> None:
        """删除集合（重建或清理时用）。"""
        if self.has_collection():
            self.client().drop_collection(collection_name=self.collection_name)
