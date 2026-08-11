"""记忆工具：把记忆系统作为 Tool 暴露给 LLM 自主调用。

LLM 通过一个统一工具对工作/情景/语义记忆执行 add / retrieve / forget / stats，
无需为每类记忆单独建工具——记忆本质上就是「增/检索/遗忘」三件事。

用法：
    tool = MemoryTool(manager)
    agent.registry.register(tool)
之后 LLM 即可像调用 calculator 一样调用记忆工具。
"""

from typing import Any, Dict

from Agent.LLMClient.tool import Tool
from Agent.Memory.manager import MemoryManager


class MemoryTool(Tool):
    """统一记忆管理工具，供 LLM 调用。"""

    def __init__(self, manager: MemoryManager = None,
                 user_id: str = "default", session_id: str = "default"):
        # 方案 A：身份在构造时烙入，LLM 的 function 参数里没有这两个字段，
        # 因此 LLM 无法伪造 user_id / session_id。
        self.user_id = user_id
        self.session_id = session_id
        self.manager = manager or MemoryManager(user_id=user_id, session_id=session_id)

    @property
    def name(self) -> str:
        return "memory"

    @property
    def description(self) -> str:
        return (
            "记忆管理工具。对工作记忆(短期)/情景记忆/语义记忆执行统一操作。"
            "action 可选: add(添加记忆)、retrieve(检索记忆)、forget(按ID遗忘一条)、stats(查看统计)。"
            "memory_type 可选: working(默认)/episodic/semantic/all(检索时可跨类型)。"
        )

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["add", "retrieve", "forget", "stats"],
                    "description": "要执行的记忆操作",
                },
                "memory_type": {
                    "type": "string",
                    "enum": ["working", "episodic", "semantic", "all"],
                    "description": "记忆类型，默认 working。retrieve 时可用 all 跨类型检索",
                },
                "content": {"type": "string", "description": "add 时记忆内容"},
                "query": {"type": "string", "description": "retrieve 时的检索关键词"},
                "memory_id": {"type": "string", "description": "forget 时要遗忘的记忆ID"},
                "limit": {"type": "integer", "description": "retrieve 返回数量，默认5"},
                "importance": {"type": "number", "description": "add 时重要性0-1，默认0.5"},
            },
            "required": ["action"],
        }

    def run(self, **kwargs) -> str:
        action = kwargs.get("action")
        memory_type = kwargs.get("memory_type") or "working"
        try:
            if action == "add":
                content = kwargs.get("content")
                if not content:
                    return "❌ add 需要 content 参数"
                mid = self.manager.add(
                    content=content,
                    memory_type=memory_type,
                    importance=kwargs.get("importance", 0.5),
                )
                return f"✅ 已添加{memory_type}记忆 (ID: {mid[:8]}...)"

            if action == "retrieve":
                query = kwargs.get("query")
                limit = kwargs.get("limit", 5)
                results = self.manager.retrieve(query=query, memory_type=memory_type, limit=limit)
                if not results:
                    return f"🔍 未找到与 '{query}' 相关的{memory_type}记忆"
                lines = [f"🔍 找到 {len(results)} 条{memory_type}记忆:"]
                for i, m in enumerate(results, 1):
                    preview = m.content[:80] + "..." if len(m.content) > 80 else m.content
                    lines.append(f"{i}. [{m.id[:8]}] {preview} (重要性:{m.importance:.2f})")
                return "\n".join(lines)

            if action == "forget":
                mid = kwargs.get("memory_id")
                if not mid:
                    return "❌ forget 需要 memory_id 参数"
                ok = self.manager.forget(mid, memory_type=memory_type)
                return f"✅ 已遗忘记忆 {mid[:8]}..." if ok else f"⚠️ 未找到ID {mid[:8]}... 的{memory_type}记忆"

            if action == "stats":
                return self._format_stats(self.manager.stats(
                    memory_type=kwargs.get("memory_type")))

            return f"❌ 不支持的操作: {action}。支持: add/retrieve/forget/stats"

        except Exception as e:
            return f"❌ 记忆操作失败: {e}"

    def _format_stats(self, stats: Dict[str, Any]) -> str:
        lines = ["📈 记忆系统统计:"]
        for mtype, s in stats.items():
            lines.append(
                f"- {mtype}: {s.get('count', 0)} 条, "
                f"tokens {s.get('current_tokens', 0)}/{s.get('max_tokens', '?')}, "
                f"平均重要性 {s.get('avg_importance', 0):.2f}"
            )
        return "\n".join(lines)
