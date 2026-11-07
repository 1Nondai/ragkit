"""评测集：把"好是什么"固化成数据。"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class EvalCase:
    """一条评测用例。"""

    question: str
    expected_keywords: list[str] = field(default_factory=list)
    reference_answer: str = ""
    kb: str | None = None
    note: str = ""
    category: str = "fact"
    image_gold: bool = False

    def to_dict(self) -> dict[str, Any]:
        """序列化（写回 jsonl 时用）。"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EvalCase:
        """反序列化；缺字段用默认值兜底，旧评测集不用改。"""
        return cls(
            question=str(data.get("question", "")).strip(),
            expected_keywords=list(data.get("expected_keywords", []) or []),
            reference_answer=str(data.get("reference_answer", "") or ""),
            kb=data.get("kb"),
            note=str(data.get("note", "") or ""),
            category=str(data.get("category", "") or "fact"),
            image_gold=bool(data.get("image_gold", False)),
        )

def load_cases(path: Path | str) -> list[EvalCase]:
    """从 JSONL 或 JSON 数组读取评测集。"""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"评测集不存在: {path}")

    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []

    if text.startswith("["):
        return [EvalCase.from_dict(item) for item in json.loads(text)]

    cases = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("//"):
            continue
        cases.append(EvalCase.from_dict(json.loads(line)))
    return cases

def save_cases(cases: Iterable[EvalCase], path: Path | str) -> Path:
    """写回 JSONL（一行一条，便于 git diff 与人工增删）。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for case in cases:
            fh.write(json.dumps(case.to_dict(), ensure_ascii=False) + "\n")
    return path
