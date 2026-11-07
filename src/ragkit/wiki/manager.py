"""Wiki 目录管理：写入、索引、日志、lint、命中统计、草稿写回。"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Sequence
from datetime import date
from pathlib import Path

from ragkit.logging_setup import get_logger
from ragkit.wiki.schema import WikiPage, safe_slug

log = get_logger("wiki.manager")

WIKI_DIRS = ("concepts", "entities", "comparisons", "queries", "drafts")
_STATS_FILE = ".stats.json"

class WikiManager:
    """wiki 目录的读写门面。"""

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)
        self.concepts_dir = self.root / "concepts"
        self.drafts_dir = self.root / "drafts"

    def ensure_dirs(self) -> None:
        """建好 concepts/entities/comparisons/queries/drafts 五个目录。"""
        for name in WIKI_DIRS:
            (self.root / name).mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _atomic_write(path: Path, text: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, path)

    def save(self, page: WikiPage, *, to_drafts: bool = False) -> Path:
        """保存单页（默认进 concepts/，草稿进 drafts/）。"""
        target_dir = self.drafts_dir if to_drafts else self.concepts_dir
        path = target_dir / f"{page.slug}.md"
        self._atomic_write(path, page.to_markdown())
        return path

    def build(self, pages: Sequence[WikiPage], *, clean: bool = True) -> dict[str, object]:
        """全量重建：清掉旧页 → 写新页 → 重写 index.md → 追加 log。"""
        self.ensure_dirs()
        if clean:
            for path in self.concepts_dir.glob("*.md"):
                path.unlink()

        for page in pages:
            self.save(page)

        self._atomic_write(self.root / "index.md", self.render_index(pages))
        self._atomic_write(self.root / "SCHEMA.md", self.render_schema())
        self.append_log("create", f"编译 {len(pages)} 页", "来源: 结构化编译（chunk → 页面）")
        return {"pages": len(pages), "root": str(self.root)}

    def render_index(self, pages: Sequence[WikiPage] | None = None) -> str:
        """生成 index.md（按类型分组的页面索引）。"""
        pages = list(pages if pages is not None else self.load())
        lines = [
            "# Wiki 索引",
            "",
            f"> 自动生成。共 {len(pages)} 页。按首字分组，每条一行：链接 + 一句话摘要。",
            "> 读这一页就能找到要看的页面，不用翻目录。",
            "",
        ]
        buckets: dict[str, list[WikiPage]] = {}
        for page in sorted(pages, key=lambda p: p.title.lower()):
            first = page.title.strip()[:1].upper() or "#"
            buckets.setdefault(first, []).append(page)
        for first in sorted(buckets):
            lines.append(f"## {first}")
            lines.append("")
            for page in buckets[first]:
                flag = " ⚠️" if page.contested else ""
                lines.append(f"- [[{page.title}]]{flag} — {page.summary()}")
            lines.append("")
        return "\n".join(lines)

    def render_schema(self) -> str:
        """生成 SCHEMA.md（页面字段与编写规范，给人看的）。"""
        return (
            "# Wiki Schema\n\n"
            "## 域\n本项目的产品/技术知识库（由 RAG 语料编译）。\n\n"
            "## 约定\n"
            "- 文件名 = 标题 slug（中文保留，空白转 `-`）\n"
            "- 每页带 frontmatter：title/slug/type/created/updated/confidence/contested/volatile/tags/sources\n"
            "- 每页至少 2 条 `[[wikilink]]`，不允许孤儿页\n"
            "- 新增页面必须同步 index.md；每次改动追加 log.md\n"
            "- 机器产出先进 `drafts/`，人工校订后才进 `concepts/`\n\n"
            "## 质量信号\n"
            "- `confidence: low` → 禁止直答（会被路由的第二道门拦下）\n"
            "- `contested: true` → 有未解冲突，退回检索\n"
            "- `volatile: true` + 超过 90 天未更新 → 过期，退回检索\n"
        )

    def append_log(self, action: str, subject: str, detail: str = "") -> None:
        """往 log.md 追加一行操作记录（谁在什么时候改了什么）。"""
        path = self.root / "log.md"
        if not path.exists():
            header = (
                "# Wiki 日志\n\n"
                "> 追加式记录。格式：`## [YYYY-MM-DD] action | subject`\n"
                "> actions: create / ingest / update / query / lint / archive\n\n"
            )
            self._atomic_write(path, header)
        entry = f"## [{date.today().isoformat()}] {action} | {subject}\n"
        if detail:
            entry += f"- {detail}\n"
        with path.open("a", encoding="utf-8") as fh:
            fh.write("\n" + entry)

    def load(self, *, include_drafts: bool = False) -> list[WikiPage]:
        """读取正式页面（lint / 统计都**不该**把 drafts 算进来）。"""
        pages: list[WikiPage] = []
        dirs = [self.concepts_dir, self.root / "entities", self.root / "comparisons",
                self.root / "queries"]
        if include_drafts:
            dirs.append(self.drafts_dir)
        for directory in dirs:
            if not directory.exists():
                continue
            for path in sorted(directory.glob("*.md")):
                try:
                    pages.append(WikiPage.from_markdown(path.stem, path.read_text(encoding="utf-8")))
                except Exception as exc:  # noqa: BLE001
                    log.warning("页面解析失败，已跳过: %s (%s)", path.name, exc)
        return pages

    def chunk_page_index(self) -> dict[str, str]:
        """chunk_id → 页面标题 的反向索引（检索结果挂页面用）。"""
        index: dict[str, str] = {}
        for page in self.load():
            for src in page.sources:
                if "#" in src:
                    index[src.rsplit("#", 1)[-1]] = page.title
        return index

    def lint(self) -> dict[str, list[str]]:
        """八条检查，每条都对应一个处理动作（只报不改的 lint 会变成告警疲劳）。"""
        pages = self.load()
        titles = {p.title for p in pages}
        inbound: dict[str, int] = {p.title: 0 for p in pages}

        findings: dict[str, list[str]] = {
            "missing_frontmatter": [],
            "broken_link": [],
            "orphan": [],
            "low_confidence": [],
            "stale_volatile": [],
            "empty_sources": [],
            "too_long": [],
            "duplicate_title": [],
        }

        seen_titles: set[str] = set()
        for page in pages:
            if page.missing_fields():
                findings["missing_frontmatter"].append(f"{page.slug}: {page.missing_fields()}")
            for link in page.links:
                if link in titles:
                    inbound[link] = inbound.get(link, 0) + 1
                else:
                    findings["broken_link"].append(f"{page.slug} -> {link}")
            if page.confidence == "low":
                findings["low_confidence"].append(page.slug)
            if page.is_stale(date.today().isoformat()):
                findings["stale_volatile"].append(page.slug)
            if not page.sources:
                findings["empty_sources"].append(page.slug)
            if len(page.body.splitlines()) > 200:
                findings["too_long"].append(page.slug)
            if page.title in seen_titles:
                findings["duplicate_title"].append(page.title)
            seen_titles.add(page.title)

        for page in pages:
            if inbound.get(page.title, 0) == 0:
                findings["orphan"].append(page.slug)

        self.append_log("lint", f"{sum(len(v) for v in findings.values())} 个问题",
                        ", ".join(f"{k}={len(v)}" for k, v in findings.items() if v))
        return findings

    @property
    def stats_path(self) -> Path:
        """命中统计文件路径（.stats.json）。"""
        return self.root / _STATS_FILE

    def stats(self) -> dict[str, object]:
        """读命中统计（文件不存在时返回空结构）。"""
        if not self.stats_path.exists():
            return {"total": 0, "hits": 0, "hit_rate": 0.0, "by_slug": {}}
        try:
            data = json.loads(self.stats_path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            return {"total": 0, "hits": 0, "hit_rate": 0.0, "by_slug": {}}
        total = int(data.get("total", 0))
        hits = int(data.get("hits", 0))
        data["hit_rate"] = round(hits / total, 4) if total else 0.0
        return data

    def record_query(self, hit: bool, slug: str | None = None, *, reset: bool = False) -> None:
        """记录一次路由结果（命中率是"这层值不值"的唯一硬证据）。"""
        data = {"total": 0, "hits": 0, "by_slug": {}} if reset else dict(self.stats())
        data.pop("hit_rate", None)
        data["total"] = int(data.get("total", 0)) + 1
        if hit:
            data["hits"] = int(data.get("hits", 0)) + 1
            if slug:
                by_slug = data.setdefault("by_slug", {})
                by_slug[slug] = int(by_slug.get(slug, 0)) + 1
        self._atomic_write(self.stats_path, json.dumps(data, ensure_ascii=False, indent=2))

    def writeback(
        self,
        question: str,
        answer: str,
        *,
        score: float,
        threshold: float = 0.7,
        min_chars: int = 40,
        sources: Sequence[str] | None = None,
    ) -> Path | None:
        """把一条高价值问答沉淀成草稿页。"""
        text = (answer or "").strip()
        if not text or text.startswith("未在知识库") or "无法作答" in text[:30]:
            return None
        if len(text) < min_chars:
            return None
        if float(score) < threshold:
            return None

        slug = safe_slug(question, max_len=40)
        for existing in self.load(include_drafts=True):
            if _similar(existing.title, question) or _similar(existing.body[:200], text[:200]):
                log.info("写回跳过：与已有页面重复（%s）", existing.slug)
                return None

        today = date.today().isoformat()
        page = WikiPage(
            slug=slug,
            title=question.strip()[:60],
            body=f"{text}\n\n## 出处\n\n" + "\n".join(f"- {s}" for s in (sources or []))
            if sources
            else text,
            type="query",
            tags=["query", "draft"],
            sources=list(sources or []),
            created=today,
            updated=today,
            confidence="medium",
        )
        self.ensure_dirs()
        path = self.save(page, to_drafts=True)
        self.append_log("writeback", page.title, f"score={score:.2f} → drafts/{path.name}")
        return path

def _similar(a: str, b: str) -> bool:
    """去重用的相似判断：先做包含，再退回 bigram Jaccard。"""
    a, b = (a or "").strip().lower(), (b or "").strip().lower()
    if not a or not b:
        return False
    if a in b or b in a:
        return True
    ga = {a[i:i + 2] for i in range(len(a) - 1)}
    gb = {b[i:i + 2] for i in range(len(b) - 1)}
    if not ga or not gb:
        return False
    return len(ga & gb) / len(ga | gb) >= 0.7

def parse_wikilinks(text: str) -> list[str]:
    """抽取页面正文里的 [[链接]]，用于建关系图。"""
    return re.findall(r"\[\[([^\]]+)\]\]", text or "")
