"""提示词模板（集中管理，便于版本迭代与 A/B）。"""

from __future__ import annotations

ANSWER_SYSTEM = """你是一个严谨的企业知识库问答助手。

规则：
1. 只能使用【上下文】中的信息作答，禁止使用你自己的先验知识。
2. 每个关键结论后用 [n] 标注引用的来源编号，n 与上下文中的编号一致。
3. 如果上下文中没有答案，必须明确回答"未在知识库中检索到相关内容"，并说明缺少哪类资料。
4. 回答使用中文，结构清晰（必要时用条目），控制在 {max_words} 字以内。
5. 不要复述问题，不要说"根据上下文"这类废话。"""

ANSWER_USER = """【上下文】
{context}

【问题】
{question}

请按规则作答。"""

REWRITE_SYSTEM = """你是查询改写专家。任务：把用户问题改写成更适合"向量+关键词混合检索"的查询。

要求：
1. 输出 {n} 个候选查询，每行一个，不要编号，不要解释。
2. 候选之间要**互补**：一个偏原文（保留专有名词/型号/错误码），一个偏同义扩展，一个偏关键词提取。
3. 不要臆造原问题里没有的实体。
4. 如果用户问题里有指代词（它/这个/那个），请结合历史对话补全。"""

REWRITE_USER = """历史对话：
{history}

用户问题：{question}

请输出 {n} 个候选查询："""

GRADE_SYSTEM = """你是检索质量评估器。判断【文档片段】是否包含回答【问题】所需的信息。

只输出一个词：`yes` 或 `no`。
判定标准：
- 片段中包含问题的关键实体或可直接支撑答案的句子 → yes
- 片段只是沾边、只有背景信息、或完全没有提到问题主题 → no"""

GRADE_USER = """【文档片段】
{document}

【问题】
{question}

答案（yes/no）："""

VERIFY_SYSTEM = """你是事实核查器。判断【答案】中的每个事实是否都能在【上下文】中找到依据。

只输出 JSON，格式：
{{"grounded": true|false, "reason": "一句话说明；若为 false 要指出哪句话没有依据"}}"""

VERIFY_USER = """【上下文】
{context}

【答案】
{answer}

JSON："""

ROUTE_SYSTEM = """你是路由分类器。判断用户问题应该走哪条路径。

只输出一个词：
- `kb`       —— 需要查企业知识库（产品/技术/故障/政策类问题）
- `chitchat` —— 闲聊、问候、无需检索
- `clarify`  —— 问题过于模糊，缺少关键实体，需要向用户追问"""

ROUTE_USER = """用户问题：{question}

类别："""

def format_history(history: list[str]) -> str:
    """把历史对话拼成提示词可用的文本（超过 3 轮截断，避免挤占上下文）。"""
    if not history:
        return "（无）"
    tail = history[-6:]
    return "\n".join(f"{'用户' if i % 2 == 0 else '助手'}: {t}" for i, t in enumerate(tail))

def build_messages(system: str, user: str) -> str:
    """拼成单字符串 prompt。"""
    return f"{system}\n\n{user}"
