# 模块化 RAG 服务 · ragkit

![CI](https://github.com/1Nondai/ragkit/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/Python-3.10%2B-3776ab)
![Milvus](https://img.shields.io/badge/VectorDB-Milvus-00a1ea)
![FastAPI](https://img.shields.io/badge/API-FastAPI-009688)
![License](https://img.shields.io/badge/License-MIT-yellow)

> 混合检索（稠密 + BM25 → RRF 融合 → 可选重排）+ Corrective/Adaptive 编排 +
> 多模态入库（PDF / 图片 / 图搜图），检索质量可量化评测，全链路带 trace 与耗时。

## 架构总图

![ragkit 全链路架构图](docs/执行流程图.jpg)

五层结构：① 数据层（ingest）→ ② 检索层（retrieval）→ ③ 编排层（graph）→ ④ 服务层（api）→ ⑤ 评测层（eval）。
真实执行顺序：`route → wiki → rewrite → retrieve → grade → generate → verify`，wiki 命中（score≥0.6 且 coverage≥0.2）直接作答短路整条检索链。

## 功能

| 能力 | 机制 | 代码位置 |
| --- | --- | --- |
| 混合检索 | Milvus HNSW 稠密 + BM25 稀疏，RRF(k=60) 融合 | `src/ragkit/retrieval/` |
| 重排 | 三档：关闭 / 词面重排（离线）/ 外部 API 重排 | `src/ragkit/retrieval/rerank.py` |
| 数据源路由 | 规则路由：知识库 / wiki / 联网兜底，可关 | `src/ragkit/graph/router.py` |
| Corrective 编排 | 生成后校验覆盖率（≥0.3），不达标改写查询重试（≤1 次） | `src/ragkit/graph/nodes.py` |
| wiki 增强层 | 语料编译成主题页，命中（score≥0.6 且 coverage≥0.2）直接作答 | `src/ragkit/wiki/` |
| 多模态入库 | PDF 解析（OCR 三档：none / rapidocr / dots）、图片裁切落盘（md5 去重）、VLM 生成描述（带缓存） | `src/ragkit/multimodal/` |
| 图搜图 | 图片向量独立字段 `dense_img`；`POST /search-image` | `src/ragkit/api/server.py` |
| 图文回答 | 命中图片块时把图（≤3 张）交给视觉模型，失败自动降级纯文本 | `src/ragkit/graph/nodes.py` |
| 增量同步 | Connector 抽象 + md5 变更检测 + JSON 断点账本；先删后写，失败重放 | `src/ragkit/connectors/` |
| ACL 权限过滤 | 文档 front-matter 声明权限组，检索前置过滤（Milvus `json_contains_any`，先过滤后算相似度） | `src/ragkit/retrieval/` + `src/ragkit/ingest/loader.py` |
| 可量化评测 | recall / hit / MRR / 覆盖率 / 延迟，逐题 A/B diff，按六类细分（fact / paraphrase / multi_hop / numeric / cross_doc / negative） | `src/ragkit/eval/` |
| 全链路观测 | 每次问答带 trace 与 timings（per-node 耗时），`--json` 可打印 | `src/ragkit/graph/state.py` |
| LLM 观测（Langfuse） | 每次问答一条 trace：节点 span、模型调用 generation、质量分数（confidence / grounded / wiki_hit / latency_ms）；评测集可导出为 dataset 跑可对比实验 | `src/ragkit/obs/` |

## 运行模式

| 模式 | 向量库 | Embedding | 生成 | 说明 |
| --- | --- | --- | --- | --- |
| 离线 | 内存库 | 哈希 | 抽句式 | 零依赖跑通全链路、CI 与单测 |
| 本地 | Docker Milvus | OpenAI 兼容接口 | DeepSeek 等 | 真实混合检索 + 生成 |
| 生产 | Milvus 集群 | 同上 | 同上 | 另需鉴权 / 网关 / 会话存储 |

离线实现与真实实现共用同一套接口（`Embeddings` / `VectorStore` Protocol），切换只改配置，不改代码。

## 快速开始

```bash
pip install -e ".[runtime]"

# 离线跑通全链路（无外部依赖）
python scripts/demo.py

# 真实模式：先起 Milvus，再入库
docker compose up -d
python scripts/ingest.py --recreate

# 语料更新后增量入库（只处理变更文件，账本在 data 目录下）
python scripts/ingest.py --incremental

python scripts/query.py "什么是 Flink Checkpoint" --json

# HTTP 服务（Swagger 在 /docs）
python scripts/serve.py
```

配置走环境变量（前缀 `RAG_`，双下划线分层，见 `.env.example`），无 `.env` 时使用默认值。

## 测试与评测

```bash
pip install -e ".[dev]"
pytest                     # 153 个用例，全离线，CI 同款
python scripts/eval.py --memory   # 产出 reports/eval_report.{md,json}
```

## 可观测性（Langfuse，可选）

内置 Langfuse 观测层（默认关闭，未启用时零开销、SDK 不加载）：

```bash
# 1. 起服务端（自托管；也可用 Langfuse Cloud）
docker compose -f docker-compose.langfuse.yml up -d   # UI: http://127.0.0.1:3000
# 2. UI 注册 → 建项目 → 拿 public/secret key → 写入 .env
#    RAG_OBS__ENABLED=true
#    RAG_OBS__PUBLIC_KEY=pk-...
#    RAG_OBS__SECRET_KEY=sk-...
# 3. 安装可选依赖后正常启动
pip install -e ".[obs]"
```

启用后每次 `/chat` 在 Langfuse 里是一条 trace：

| 层级 | 内容 | 代码 |
| --- | --- | --- |
| trace（`rag-query`） | 问题、session/user、kb、tags | `obs/langfuse_sink.py` |
| chain span | 每个图节点一个（route / wiki / retrieve / grade / generate / verify…） | `graph/graph.py` |
| generation | 每次 LLM 调用（模型名 / 输入 / 输出 / 耗时） | `obs/langfuse_sink.py` |
| scores | `confidence`（数值）、`grounded` / `wiki_hit` / `degraded`（布尔）、`latency_ms` | `obs/langfuse_sink.py` |

评测集可一键导出为 dataset 并以实验跑分，同一实验名多轮跑分自动可对比：

```bash
python scripts/langfuse_eval.py                 # 导出 dataset + 跑一轮实验
python scripts/langfuse_eval.py --run-name v2   # 改参后跑第二轮，UI 逐题对比
```

设计约束：观测是旁路不是主链路——未启用、缺 key、SDK 缺失、上报失败四种情况全部降级为无操作，不影响问答结果（CI 不安装 langfuse，持续验证该降级路径）。

## 设计要点

- **检索组件可替换**：embedding / 向量库 / 重排都是 Protocol 接口，离线实现用于测试与 CI，真实实现只差一份配置。
- **降级而非失败**：OCR 不可用回落文本层、视觉模型失败回落纯文本、路由不可用回落知识库、wiki 未命中回落检索——每一级降级都有观测点。
- **重试有上限**：Corrective 改写重试最多 1 次，避免循环放大成本。
- **可观测是旁路**：Langfuse 三层 span（trace / node / generation）+ 质量分数，任何观测故障都降级为无操作。
- **评测先行**：改检索参数前先跑 `eval_offline.py --a-b` 拿逐题 diff，不做无对照的调优；评测集 40 条按六类细分，平均值掩盖不了的"某类题全面变差"靠分类表看。
- **权限在检索前过滤**：用户组作为 Milvus search 表达式前置下发，不召回后筛（后筛会把 top_k 筛空）；内存实现与 Milvus 表达式语义对齐，单测防行为分叉。

## 已知限制

- 离线档（哈希 embedding）只验证流程正确性，中文语义命中质量差，不能评估检索质量。
- `dashscope-mm` 图文 embedding 按官方接口实现，尚未接入真实 Key 验证；接入后应先用 `data/eval/eval_set_multimodal.jsonl` 跑 `ragkit-eval`。
- 会话历史存内存，重启即失；生产需外接 Redis。
- ACL 为单租户简化版（文档级权限组），未做行级权限、限流与多租户隔离。
