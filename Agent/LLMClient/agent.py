"""Agent 基类。

仿 MyAgent/chapter7/core/agent.py 的设计模式，但自包含：不依赖
chapter7 的 Config / Messages，历史用轻量 Message dataclass 承载，
与 OpenAI 兼容的 {"role","content"} 消息格式互转方便。
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class Message:
    """单条对话历史记录（跨轮上下文用）。"""
    role: str
    content: str

    def to_dict(self) -> dict:
        return {"role": self.role, "content": self.content}


class Agent(ABC):
    """Agent 基类。

    持有 name / llm / system_prompt / config / _history，定义抽象 run()，
    提供跨轮历史的增删查便利方法。子类负责实现具体的运行循环。
    """

    def __init__(
        self,
        name: str,
        llm,
        system_prompt: Optional[str] = None,
        config=None,
    ):
        self.name = name
        self.llm = llm
        self.system_prompt = system_prompt
        self.config = config
        self._history: List[Message] = []

    @abstractmethod
    def run(self, input_text: str, **kwargs) -> str:
        """运行 Agent，返回最终文本响应。"""
        ...

    def add_message(self, *args) -> None:
        """追加一条历史记录。支持三种调用：
        - add_message(Message(role="user", content="..."))
        - add_message({"role": "user", "content": "..."})
        - add_message("user", "...")
        """
        if len(args) == 1:
            m = args[0]
            if isinstance(m, Message):
                self._history.append(m)
            elif isinstance(m, dict):
                self._history.append(Message(role=m["role"], content=m["content"]))
            else:
                raise TypeError("单参数形式需传入 Message 或 dict")
        elif len(args) == 2:
            self._history.append(Message(role=args[0], content=args[1]))
        else:
            raise TypeError("add_message 用法：add_message(msg) 或 add_message(role, content)")

    def clear_history(self) -> None:
        """清空历史记录。"""
        self._history.clear()

    def get_history(self) -> List[Message]:
        """获取历史记录副本。"""
        return self._history.copy()

    def __str__(self) -> str:
        return f"Agent(name={self.name}, model={getattr(self.llm, 'model', '?')})"

    def __repr__(self) -> str:
        return self.__str__()
