"""文档加载：把原始文件变成"带标题路径的段落"。"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
_FENCE_RE = re.compile(r"^\s*(```|~~~)")

SUPPORTED_SUFFIX = {".md", ".markdown", ".txt"}

@dataclass
class Section:
    """一个"标题下的连续正文块"。"""

    text: str
    title_path: list[str] = field(default_factory=list)
    level: int = 0
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def title(self) -> str:
        """用 ' -> ' 连接标题路径，作为引用里显示的标题。"""
        return " -> ".join(self.title_path) if self.title_path else ""

    def is_empty(self) -> bool:
        return not self.text.strip()

@dataclass
class LoadedDocument:
    """一个文件的解析结果。"""

    source: str
    filename: str
    filetype: str
    sections: list[Section] = field(default_factory=list)

    @property
    def section_count(self) -> int:
        return len(self.sections)

def _strip_fences(lines: Iterable[str]) -> list[str]:
    """丢掉 ``` 代码块内容：因为代码块会严重稀释 embedding 的语义密度。"""
    out: list[str] = []
    in_fence = False
    for line in lines:
        if _FENCE_RE.match(line):
            in_fence = not in_fence
            continue
        if not in_fence:
            out.append(line)
    return out

def parse_markdown(text: str, *, drop_code_blocks: bool = True) -> list[Section]:
    """解析 Markdown 文本为 Section 列表（维护标题栈）。"""
    lines = text.splitlines()
    if drop_code_blocks:
        lines = _strip_fences(lines)

    sections: list[Section] = []
    title_stack: list[str] = []
    buffer: list[str] = []
    current_level = 0

    def flush() -> None:
        body = "\n".join(buffer).strip()
        if body:
            sections.append(
                Section(text=body, title_path=list(title_stack), level=current_level)
            )
        buffer.clear()

    for line in lines:
        m = _HEADING_RE.match(line)
        if m:
            flush()
            level = len(m.group(1))
            title = m.group(2).strip().strip("#").strip()
            while len(title_stack) >= level:
                title_stack.pop()
            title_stack.append(title)
            current_level = level
            continue
        buffer.append(line)

    flush()

    if not sections:
        body = "\n".join(lines).strip()
        if body:
            sections.append(Section(text=body, title_path=[], level=0))
    return sections

def load_file(path: Path, *, drop_code_blocks: bool = True) -> LoadedDocument:
    """加载单个文件。未知后缀按纯文本处理。"""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"文件不存在: {path}")
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_SUFFIX:
        raise ValueError(f"暂不支持的文件类型: {suffix}（支持 {sorted(SUPPORTED_SUFFIX)}）")

    text = path.read_text(encoding="utf-8", errors="ignore")
    sections = parse_markdown(text, drop_code_blocks=drop_code_blocks) if suffix in {
        ".md", ".markdown"
    } else [Section(text=text.strip(), title_path=[], level=0)]

    for s in sections:
        s.metadata.update(
            {"source": str(path), "filename": path.name, "filetype": suffix.lstrip(".")}
        )
    return LoadedDocument(
        source=str(path), filename=path.name, filetype=suffix.lstrip("."), sections=sections
    )

def iter_files(data_dir: Path, suffixes: Iterable[str] | None = None) -> list[Path]:
    """递归扫描目录，返回按名称排序的文件列表（顺序稳定 = 入库可复现）。"""
    allow = {s.lower() for s in (suffixes or SUPPORTED_SUFFIX)}
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"数据目录不存在: {data_dir}")
    return sorted(p for p in data_dir.rglob("*") if p.is_file() and p.suffix.lower() in allow)

def load_directory(data_dir: Path, **kwargs) -> list[LoadedDocument]:
    """加载目录下所有支持的文件；单文件失败只记录不中断（批量入库的基本要求）。"""
    from ragkit.logging_setup import get_logger

    log = get_logger("ingest.loader")
    docs: list[LoadedDocument] = []
    for path in iter_files(data_dir):
        try:
            docs.append(load_file(path, **kwargs))
        except Exception as exc:  # noqa: BLE001
            log.warning("加载失败，已跳过: %s (%s)", path.name, exc)
    return docs
