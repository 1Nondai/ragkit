"""LLM 抽象：在线模型 + 离线兜底。"""

from __future__ import annotations

import logging
from typing import Protocol, runtime_checkable

logger = logging.getLogger("ragkit.llm")

@runtime_checkable
class ChatModel(Protocol):
    """聊天模型接口：invoke 收单字符串，invoke_messages 收多模态消息。"""
    name: str

    def invoke(self, prompt: str) -> str: ...

    def invoke_messages(self, messages: list[dict]) -> str: ...

def supports_vision(chat: object) -> bool:
    """这个 chat 实现能不能吃多模态消息（图 + 文字）。"""
    return not isinstance(chat, ExtractiveLLM) and callable(
        getattr(chat, "invoke_messages", None)
    )

class OpenAICompatChat:
    """OpenAI 兼容聊天接口（DeepSeek / 通义 / 智谱 / OpenAI 都可用）。"""

    def __init__(
        self,
        model: str,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        temperature: float = 0.2,
        timeout: float = 60.0,
        max_retries: int = 2,
    ) -> None:
        self.name = model
        self.base_url = base_url
        self.api_key = api_key
        self.temperature = temperature
        self.timeout = timeout
        self.max_retries = max_retries
        self._client = None

    def _ensure(self):
        if self._client is None:
            from langchain_openai import ChatOpenAI

            kwargs = {
                "model": self.name,
                "temperature": self.temperature,
                "timeout": self.timeout,
                "max_retries": self.max_retries,
            }
            if self.base_url:
                kwargs["base_url"] = self.base_url
            if self.api_key:
                kwargs["api_key"] = self.api_key
            self._client = ChatOpenAI(**kwargs)
        return self._client

    def invoke(self, prompt: str) -> str:
        """纯文本调用（大多数节点用这条）。"""
        resp = self._ensure().invoke(prompt)
        return getattr(resp, "content", str(resp))

    def invoke_messages(self, messages: list[dict]) -> str:
        """多模态消息调用：messages 是 OpenAI 格式（content 可为数组，
        内含 image_url 块）。生成答案时命中图片块就走这条路。
        """
        resp = self._ensure().invoke(messages)
        return getattr(resp, "content", str(resp))

class ExtractiveLLM:
    """离线兜底：从上下文中抽取最相关句子拼答案（无生成能力）。"""

    def __init__(self, max_sentences: int = 3) -> None:
        self.name = "extractive-offline"
        self.max_sentences = max_sentences

    @staticmethod
    def _terms(text: str) -> set:
        import re
        terms = set(re.findall(r"[a-z0-9]+", (text or "").lower()))
        terms |= {
            text[i : i + 2]
            for i in range(len(text or "") - 1)
            if all("\u4e00" <= ch <= "\u9fff" for ch in text[i : i + 2])
        }
        return {t for t in terms if t.strip()}

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        import re

        parts = re.split(r"(?<=[。！？.!?;；\n])", text or "")
        return [p.strip() for p in parts if p and p.strip()]

    def invoke(self, prompt: str) -> str:
        """纯文本调用（大多数节点用这条）。"""
        context, question = prompt, ""
        for sep in ("【问题】", "[问题]"):
            if sep in prompt:
                ctx_part, question = prompt.split(sep, 1)
                for ctx_sep in ("【上下文】", "[上下文]"):
                    if ctx_sep in ctx_part:
                        context = ctx_part.split(ctx_sep, -1)[-1]
                        break
                break

        q_terms = self._terms(question)
        sentences = self._split_sentences(context)
        scored = []
        for s in sentences:
            if s.startswith("[") and "来源" in s[:20]:
                continue
            overlap = len(q_terms & self._terms(s)) / (len(q_terms) or 1)
            scored.append((overlap, s))
        scored.sort(key=lambda x: x[0], reverse=True)

        picked = [s for _, s in scored[: self.max_sentences]]
        if not picked:
            return "未检索到可用于作答的内容（离线模式，无生成能力）。"
        body = "\n".join(f"- {s}" for s in picked)
        return (
            f"{body}\n\n"
            "> 说明：当前为**离线兜底模式**，未连接大模型，"
            "以上为从检索上下文抽取的原文片段，非生成式答案。"
        )

    def invoke_messages(self, messages: list[dict]) -> str:
        """离线兜底的多模态调用：把消息里的文字抽出来，图片直接忽略。"""
        parts: list[str] = []
        for m in messages:
            content = m.get("content", "")
            if isinstance(content, str):
                parts.append(content)
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        parts.append(block.get("text", ""))
        return self.invoke("\n".join(parts))

def build_chat_model():
    """按配置构造模型；不可用时回落到离线抽取模型（打 WARNING）。"""
    from ragkit.config import get_settings

    s = get_settings().llm
    if not s.api_key:
        logger.warning("未配置 LLM_API_KEY，使用离线抽取模型（无生成能力）")
        return ExtractiveLLM()
    try:
        return OpenAICompatChat(
            model=s.model,
            base_url=s.base_url,
            api_key=s.api_key,
            temperature=s.temperature,
            timeout=s.timeout,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("LLM 初始化失败，回落到离线模型: %s", exc)
        return ExtractiveLLM()
