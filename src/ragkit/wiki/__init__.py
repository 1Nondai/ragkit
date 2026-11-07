"""Wiki 层：把 chunk 编译成可复用、可溯源、可校订的知识页（RAG 的可选增强层）。"""

from ragkit.wiki.compiler import compile_from_chunks, compile_pages
from ragkit.wiki.manager import WikiManager
from ragkit.wiki.router import WikiDecision, WikiRouter, score_page
from ragkit.wiki.schema import WikiPage, safe_slug

__all__ = [
    "WikiPage",
    "safe_slug",
    "compile_pages",
    "compile_from_chunks",
    "WikiManager",
    "WikiRouter",
    "WikiDecision",
    "score_page",
]
