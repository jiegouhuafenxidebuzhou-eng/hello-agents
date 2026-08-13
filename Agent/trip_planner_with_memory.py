"""带记忆系统的旅游规划 Agent —— 对照 chapter13 的 trip_planner_agent 写的示例。

chapter13 的 trip_planner_agent 只做了两件事：
1. 用 SimpleAgent 挂 MCP(高德) 工具做多 Agent 分工；
2. 手工把上一步输出拼进下一步 prompt。

它**没有记忆**：每次 plan_trip 都是干净的、不记得用户上一次去过哪、偏好什么。

本文件在它基础上补上「记忆系统」(Agent/Memory/manager.py 的 MemoryManager)：
- 工作记忆 (working)：本会话内的短期上下文，纯内存、有容量上限、会遗忘；
- 情景记忆 (episodic)：跨会话持久化 (本地 SQLite，无 Qdrant 时自动降级为关键词检索)，
  用来记住「这个用户喜欢历史文化、上次去了北京」这类长期事实。

记忆的两个挂钩点：
- 规划前：retrieve() 召回相关记忆，塞进 system prompt → Agent 不再失忆；
- 规划后：add() 把本次的用户偏好 / 行程结果写回记忆 → 下次能用上。

运行依赖：
- LLM：在 Agent/Memory/.env 或 Agent/LLMClient/.env 配好 LLM_MODEL_ID / LLM_API_KEY / LLM_BASE_URL；
- MCP(高德) 工具：可选。配了 AMAP_MAPS_API_KEY 才挂工具，没配就走纯 LLM 规划路径，
  记忆系统照样演示。所以没有高德 key 也能跑通本示例。
"""

import os
import json
from typing import Optional

from hello_agents import SimpleAgent, HelloAgentsLLM
from hello_agents.tools import MCPTool

from Agent.Memory.manager import MemoryManager


# ============ Agent 提示词 ============

PLANNER_PROMPT = """你是行程规划专家。根据用户需求和已知信息，生成一份简要的旅行计划。

要求：
1. 每天安排 2-3 个景点，给出早中晚三餐建议；
2. 如果「记忆」里提到用户偏好（喜欢的景点类型、去过的城市、预算倾向），务必优先满足；
3. 输出用简洁的中文列表/短段落即可，不需要 JSON。

**如果挂载了高德地图工具**，需要真实景点/天气/酒店信息时按下面格式调用：
`[TOOL_CALL:amap_maps_text_search:keywords=关键词,city=城市名]`
"""


class MemoryAwareTripPlanner:
    """带记忆的旅游规划 Agent。

    对照 chapter13 的 MultiAgentTripPlanner：
    - 保留 SimpleAgent + MCP 工具 的用法（和原版一致）；
    - 新增 MemoryManager：规划前召回、规划后写入。
    """

    def __init__(self, user_id: str = "default", use_mcp: bool = True):
        print("🔄 初始化带记忆的旅游规划 Agent ...")
        self.user_id = user_id
        self.session_id = f"trip-{user_id}"

        # ---- LLM（hello_agents 的实现带 .invoke，SimpleAgent 需要）----
        self.llm = HelloAgentsLLM()

        # ---- 记忆系统：工作记忆(本会话) + 情景记忆(跨会话, 本地 SQLite) ----
        # 身份 user_id 在构造时烙入，所有读写都带 user_id 过滤，LLM 无法越权。
        self.memory = MemoryManager(
            user_id=user_id,
            session_id=self.session_id,
            enable_episodic=True,   # 开启跨会话持久化（无 Qdrant 自动降级为关键词检索）
            enable_semantic=False,  # 语义/图谱记忆本示例不启用，保持简单
        )
        print(f"   记忆后端: {self.memory.enabled_types}")

        # ---- 规划 Agent（和 chapter13 一样用 SimpleAgent）----
        self.planner = SimpleAgent(
            name="行程规划专家",
            llm=self.llm,
            system_prompt=PLANNER_PROMPT,
        )

        # ---- 可选：挂高德 MCP 工具（和 chapter13 一致）----
        # 没配 AMAP_MAPS_API_KEY 就跳过，Agent 退化为纯 LLM 规划，记忆照样工作。
        self._mcp_ready = False
        if use_mcp:
            api_key = os.getenv("AMAP_MAPS_API_KEY")
            if api_key:
                self.amap = MCPTool(
                    name="amap",
                    description="高德地图服务",
                    server_command=["uvx", "amap-mcp-server"],
                    env={"AMAP_MAPS_API_KEY": api_key},
                    auto_expand=True,
                )
                self.planner.add_tool(self.amap)
                self._mcp_ready = True
                print(f"   已挂载高德 MCP 工具，可用工具 {len(self.planner.list_tools())} 个")
            else:
                print("   未配置 AMAP_MAPS_API_KEY，跳过 MCP 工具（纯 LLM 规划路径）")

        print("✅ 初始化完成\n")

    # ---------- 记忆召回：规划前调用 ----------
    def _recall(self, query: str) -> str:
        """从记忆里召回与本次请求相关的内容，拼成 prompt 片段。"""
        # 工作记忆：本会话刚聊过的（短期）
        working = self.memory.retrieve(query=query, memory_type="working", limit=3)
        # 情景记忆：跨会话的长期偏好 / 历史行程
        episodic = self.memory.retrieve(query=query, memory_type="episodic", limit=3)

        if not working and not episodic:
            return "（暂无历史记忆）"

        lines = []
        if episodic:
            lines.append("【长期记忆 / 历史偏好】")
            lines.extend(f"- {m.content}" for m in episodic)
        if working:
            lines.append("【本会话近期上下文】")
            lines.extend(f"- {m.content}" for m in working)
        return "\n".join(lines)

    # ---------- 记忆写入：规划后调用 ----------
    def _remember(self, user_request: str, plan: str) -> None:
        """把本次交互写回记忆。"""
        # 1) 工作记忆：原始问答，本会话内可被下一轮召回（短期、会过期驱逐）
        self.memory.add(
            content=f"用户需求：{user_request}",
            memory_type="working",
            importance=0.6,
            metadata={"kind": "user_request"},
        )
        self.memory.add(
            content=f"规划结果摘要：{plan[:120]}",
            memory_type="working",
            importance=0.5,
            metadata={"kind": "plan_summary"},
        )

        # 2) 情景记忆：抽出可复用的长期事实（偏好/去过的城市），跨会话持久化。
        #    这里做最简单的关键词抽取；生产里可让 LLM 抽取结构化偏好。
        facts = self._extract_facts(user_request)
        for fact in facts:
            self.memory.add(
                content=fact,
                memory_type="episodic",
                importance=0.8,  # 偏好类事实重要性高，衰减更慢
                metadata={"kind": "preference", "source": user_request},
            )

    @staticmethod
    def _extract_facts(user_request: str) -> list[str]:
        """从用户请求里粗略抽取值得长期记住的事实（演示用，非生产级）。"""
        facts = []
        # 城市偏好：命中常见城市词就记一条
        cities = ["北京", "上海", "成都", "西安", "杭州", "广州", "深圳", "南京", "重庆"]
        for c in cities:
            if c in user_request:
                facts.append(f"用户曾计划前往{c}旅行")
        # 景点偏好
        if any(k in user_request for k in ["历史", "文化", "古迹", "博物馆"]):
            facts.append("用户偏好：历史文化类景点")
        if any(k in user_request for k in ["自然", "风景", "公园", "山水"]):
            facts.append("用户偏好：自然风光类景点")
        if any(k in user_request for k in ["美食", "小吃", "吃货"]):
            facts.append("用户偏好：美食优先")
        if "预算" in user_request or "便宜" in user_request or "省钱" in user_request:
            facts.append("用户偏好：经济型/省钱")
        return facts

    # ---------- 主入口 ----------
    def plan_trip(self, user_request: str) -> str:
        print(f"\n{'='*60}")
        print(f"📍 用户请求：{user_request}")
        print(f"{'='*60}")

        # 1) 规划前：召回记忆
        memory_context = self._recall(user_request)
        print(f"🧠 召回记忆：\n{memory_context}\n")

        # 2) 把记忆拼进 prompt（这是「记忆生效」的关键一步）
        prompt = (
            f"【记忆】\n{memory_context}\n\n"
            f"【本次用户需求】\n{user_request}\n\n"
            f"请结合记忆和需求给出旅行计划。"
        )
        plan = self.planner.run(prompt)

        # 3) 规划后：写回记忆（下次能用上）
        self._remember(user_request, plan)
        print(f"💾 已写入记忆（工作记忆 + 情景记忆）")

        # 4) 顺手看一眼记忆现状
        print(f"📊 记忆统计：{self.memory.stats()}")
        return plan


# ============================================================================
# 演示：两轮对话，第二轮能用到第一轮的记忆
# ============================================================================
if __name__ == "__main__":
    # 注意：同一 user_id 才能跨会话召回情景记忆。换个 id 就是另一个人的记忆。
    planner = MemoryAwareTripPlanner(user_id="alice", use_mcp=True)

    # ---- 第一轮：留下偏好 ----
    print("\n" + "#" * 60)
    print("# 第一轮")
    print("#" * 60)
    plan1 = planner.plan_trip("我想去北京玩 3 天，喜欢历史文化和博物馆，预算有限想省钱")
    print("\n📝 第一轮计划（节选）：")
    print(plan1[:300] + " ...\n")

    # ---- 第二轮：什么都不说，看 Agent 是否记得 alice 喜欢历史文化 + 去过北京 ----
    print("#" * 60)
    print("# 第二轮（模拟新会话：只说『再帮我规划一次旅行』，看记忆是否生效）")
    print("#" * 60)
    plan2 = planner.plan_trip("再帮我规划一次 3 天的旅行")
    print("\n📝 第二轮计划（节选）：")
    print(plan2[:300] + " ...\n")

    # 关键观察：第二轮的「召回记忆」里应出现
    #   - 用户曾计划前往北京旅行
    #   - 用户偏好：历史文化类景点
    #   - 用户偏好：经济型/省钱
    # 这就是 chapter13 原版做不到的「跨会话记忆」。
