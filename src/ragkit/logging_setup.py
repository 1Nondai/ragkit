"""日志配置。"""

from __future__ import annotations

import logging
import os
import sys

_CONFIGURED = False

_DEFAULT_FMT = "%(asctime)s | %(levelname)-7s | %(name)-28s | %(message)s"
_DEFAULT_DATE = "%Y-%m-%d %H:%M:%S"

def setup_logging(level: str | None = None, *, force: bool = False) -> logging.Logger:
    """初始化根 logger（幂等）。"""
    global _CONFIGURED

    if _CONFIGURED and not force:
        return logging.getLogger("ragkit")

    level_name = (level or os.environ.get("RAG_LOG_LEVEL", "INFO")).upper()
    numeric = getattr(logging, level_name, logging.INFO)

    root = logging.getLogger()
    root.setLevel(numeric)
    if force:
        for h in list(root.handlers):
            root.removeHandler(h)

    if not root.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(_DEFAULT_FMT, datefmt=_DEFAULT_DATE))
        root.addHandler(handler)

    for noisy in ("httpx", "httpcore", "openai", "urllib3", "pymilvus"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _CONFIGURED = True
    return logging.getLogger("ragkit")

def get_logger(name: str) -> logging.Logger:
    """取 logger（会顺带完成初始化）。"""
    setup_logging()
    return logging.getLogger(name if name.startswith("ragkit") else f"ragkit.{name}")
