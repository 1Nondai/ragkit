"""编译：把检索用的 chunk 重组成"知识页"。"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Iterable, Sequence
from datetime import date

from ragkit.ingest.chunker import Chunk
from ragkit.wiki.schema import WikiPage, safe_slug

Summarizer = Callable[[str, list[str]], str]

def _norm(text: str) -> str:
    return " ".join((text or "").split())

def group_key(chunk: Chunk) -> str:
    """页面分组键：优先用标题路径的叶子（最具体的一节），否则回退文件名。"""
    title = str(chunk.metadata.get("title") or "").strip()
    if not title:
        return str(chunk.metadata.get("filename") or "untitled")
    leaf = title.split("->")[-1].strip()
    return leaf or title

def compile_pages(
    chunks: Sequence[Chunk],
    *,
    today: str | None = None,
    summarizer: Summarizer | None = None,
    max_sources_per_page: int = 12,
    max_chars_per_page: int = 6000,
) -> list[WikiPage]:
    """把 chunk 编译成 Wiki 页（按主题合并、去重、建链）。"""
    today = today or date.today().isoformat()

    groups: dict[str, list[Chunk]] = defaultdict(list)
    for chunk in chunks:
        groups[group_key(chunk)].append(chunk)

    pages: list[WikiPage] = []
    for title, items in groups.items():
        seen: set[str] = set()
        texts: list[str] = []
        for chunk in items:
            key = _norm(chunk.text)
            if key in seen:
                continue
            seen.add(key)
            texts.append(chunk.text.strip())

        body_parts = texts
        if summarizer is not None:
            try:
                summary = summarizer(title, texts)
                if summary and summary.strip():
                    body_parts = [summary.strip(), "\n## 原文摘录\n\n" + "\n\n".join(texts)]
            except Exception:  # noqa: BLE001 - 摘要失败必须退回摘抄，不能让编译整体失败
                body_parts = texts

        body = "\n\n".join(body_parts)[:max_chars_per_page]

        sources = [
            f"{c.metadata.get('filename', 'unknown')}#{c.chunk_id}"
            for c in items[:max_sources_per_page]
        ]
        confidences = {c.metadata.get("filename") for c in items}
        confidence = "high" if len(confidences) >= 2 else "medium"

        pages.append(
            WikiPage(
                slug=safe_slug(title),
                title=title,
                body=body,
                type="concept",
                tags=sorted({str(c.metadata.get("category", "content")) for c in items}),
                sources=sources,
                created=today,
                updated=today,
                confidence=confidence,
                aliases=_aliases(title, items),
                volatile=_is_volatile(title, items),
            )
        )

    _link_pages(pages)
    return sorted(pages, key=lambda p: p.title)

def _aliases(title: str, items: Sequence[Chunk]) -> list[str]:
    """别名：标题里的英文/型号片段（"Milvus 是什么" → "Milvus"）。

    别名直接决定路由命中率：用户很少按页面标题原样提问，
    没有别名时"什么是 Milvus"对标题"Milvus 是什么"的相似度会低到过不了阈值。
    """
    import re

    base = title.strip()
    out: list[str] = []
    for token in re.findall(r"[A-Za-z][A-Za-z0-9_.\-]{2,}", base):
        if token.lower() not in {a.lower() for a in out}:
            out.append(token)
    stripped = re.sub(r"(是什么|是什么？|怎么|如何|为什么|哪些|多少|吗|呢|\?)", "", base).strip()
    if stripped and stripped != base:
        out.append(stripped)
    return out[:5]

def _is_volatile(title: str, items: Sequence[Chunk]) -> bool:
    """易腐化判定：版本号/价格/上限这类知识会过期。"""
    keywords = ("版本", "价格", "报价", "上限", "限制", "version", "release", "limit")
    text = title + " " + " ".join(c.text[:200] for c in items[:3])
    return any(k in text for k in keywords)

def _link_pages(pages: list[WikiPage], top_k: int = 3) -> None:
    """给每页建立 2~3 条出链（同批编译里最相关的页）。

    链接怎么选：共享英文 token 越多越相关，其次按标题字符重合。
    不够就补相邻页——**孤儿页是 wiki 的硬伤**（lint 第一条就报它）。
    """
    if len(pages) < 2:
        return

    token_sets = {}
    for page in pages:
        import re

        tokens = {t.lower() for t in re.findall(r"[A-Za-z0-9]{3,}", page.title + page.body[:600])}
        token_sets[page.slug] = tokens

    for idx, page in enumerate(pages):
        scored = []
        for other in pages:
            if other.slug == page.slug:
                continue
            overlap = len(token_sets[page.slug] & token_sets[other.slug])
            scored.append((overlap, other.title))
        scored.sort(key=lambda x: x[0], reverse=True)
        picks = [title for overlap, title in scored[:top_k] if overlap > 0]
        if len(picks) < 2:
            for offset in (1, -1, 2, -2):
                neighbour = pages[(idx + offset) % len(pages)]
                if neighbour.title not in picks and neighbour.title != page.title:
                    picks.append(neighbour.title)
                if len(picks) >= 2:
                    break
        link_block = "\n\n## 相关\n\n" + "\n".join(f"- [[{t}]]" for t in picks[:top_k])
        if page.body.find("## 相关") < 0:
            page.body = page.body.rstrip() + link_block
        page.links = picks[:top_k]

    _ensure_no_orphans(pages)

def _ensure_no_orphans(pages: list[WikiPage]) -> None:
    """第二遍：给没有入链的页面补反向链接。"""
    inbound: dict[str, int] = {p.title: 0 for p in pages}
    for page in pages:
        for link in page.links:
            if link in inbound:
                inbound[link] += 1

    for page in pages:
        if inbound.get(page.title, 0) > 0:
            continue
        for host in pages:
            if host.title == page.title:
                continue
            if page.title in host.links:
                continue
            host.body = host.body.rstrip() + f"\n- [[{page.title}]]"
            host.links = host.links + [page.title]
            inbound[page.title] = inbound.get(page.title, 0) + 1
            if inbound[page.title] >= 1:
                break

def compile_from_chunks(chunks: Iterable[Chunk], **kwargs) -> list[WikiPage]:
    """把 chunk 按标题路径分组，重组成知识页并写盘。"""
    return compile_pages(list(chunks), **kwargs)
