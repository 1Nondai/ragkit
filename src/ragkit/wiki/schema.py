"""Wiki 页面模型与 frontmatter 读写。"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

REQUIRED_FIELDS = ("title", "type", "created", "updated", "tags", "sources")

_FM_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n?(.*)$", re.S)

@dataclass
class WikiPage:
    """一个概念/实体页。"""

    slug: str
    title: str
    body: str = ""
    type: str = "concept"
    tags: list[str] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)
    created: str = ""
    updated: str = ""
    confidence: str = "high"
    contested: bool = False
    volatile: bool = False
    aliases: list[str] = field(default_factory=list)
    links: list[str] = field(default_factory=list)

    def to_markdown(self) -> str:
        """渲染成带 front-matter 的 md 页面（落盘用）。

        格式：YAML front-matter + Markdown body，便于人类阅读和 git diff。
        """
        fm = [
            "---",
            f"title: {self.title}",
            f"slug: {self.slug}",
            f"type: {self.type}",
            f"created: {self.created}",
            f"updated: {self.updated}",
            f"confidence: {self.confidence}",
            f"contested: {str(self.contested).lower()}",
            f"volatile: {str(self.volatile).lower()}",
            f"tags: [{', '.join(self.tags)}]",
            f"sources: [{', '.join(self.sources)}]",
        ]
        if self.aliases:
            fm.append(f"aliases: [{', '.join(self.aliases)}]")
        fm.append("---")
        return "\n".join(fm) + "\n\n" + self.body.strip() + "\n"

    @classmethod
    def from_markdown(cls, slug: str, text: str) -> WikiPage:
        """解析本项目写出的格式（缺失字段给默认值，不抛异常）。"""
        match = _FM_RE.match(text)
        meta: dict[str, str] = {}
        body = text
        if match:
            body = match.group(2)
            for line in match.group(1).splitlines():
                if ":" not in line:
                    continue
                key, _, value = line.partition(":")
                meta[key.strip()] = value.strip()

        def _list(raw: str) -> list[str]:
            raw = raw.strip().strip("[]")
            return [x.strip() for x in raw.split(",") if x.strip()]

        return cls(
            slug=meta.get("slug", slug),
            title=meta.get("title", slug),
            body=body.strip(),
            type=meta.get("type", "concept"),
            tags=_list(meta.get("tags", "")),
            sources=_list(meta.get("sources", "")),
            created=meta.get("created", ""),
            updated=meta.get("updated", ""),
            confidence=meta.get("confidence", "high"),
            contested=meta.get("contested", "false").lower() == "true",
            volatile=meta.get("volatile", "false").lower() == "true",
            aliases=_list(meta.get("aliases", "")),
            links=sorted(set(re.findall(r"\[\[([^\]]+)\]\]", body))),
        )

    def missing_fields(self) -> list[str]:
        """lint 用：返回缺失的必填字段（空 body、必填标签等）。"""
        missing = []
        for name in REQUIRED_FIELDS:
            if not getattr(self, name, None):
                missing.append(name)
        return missing

    def summary(self, limit: int = 80) -> str:
        """index.md 用的一句话摘要（取正文第一句，句号/问号/感叹号/换行截断）。"""
        first = re.split(r"[。！？\n]", self.body.strip(), maxsplit=1)[0]
        return (first[:limit] + "…") if len(first) > limit else first

    def is_stale(self, today: str, stale_days: int = 90) -> bool:
        """易腐化页面（volatile=True）且超过 stale_days 未更新 = 过期。

        用于第三/四道门的过期检测：过期页面不参与融合或降权。
        """
        if not self.volatile or not self.updated:
            return False
        from datetime import date

        try:
            updated = date.fromisoformat(self.updated[:10])
            now = date.fromisoformat(today[:10])
        except ValueError:
            return False
        return (now - updated).days > stale_days

def safe_slug(title: str, max_len: int = 60) -> str:
    """把标题转成文件名安全的 slug（保留中文，去掉路径分隔符与空白）。"""
    slug = re.sub(r"[\\/:*?\"<>|\s]+", "-", title.strip())
    slug = re.sub(r"-{2,}", "-", slug).strip("-")
    return (slug or "untitled")[:max_len]
