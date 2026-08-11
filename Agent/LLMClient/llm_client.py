from typing import Any, Dict, List

from dotenv import load_dotenv
import os
from pathlib import Path

from openai import OpenAI

# 明确指向脚本所在目录下的 .env，避免工作目录不同导致加载失败
load_dotenv(Path(__file__).parent / ".env")


class HelloAgentsLLM:
    """与大语言模型交互的客户端，基于 OpenAI 兼容接口封装。"""

    def __init__(self, model: str = None, apiKey: str = None, baseUrl: str = None, timeout: int = None):
        self.model = model or os.getenv("LLM_MODEL_ID")
        apiKey = apiKey or os.getenv("LLM_API_KEY")
        baseUrl = baseUrl or os.getenv("LLM_BASE_URL")
        timeout = timeout or int(os.getenv("LLM_TIMEOUT", 60))

        if not all([self.model, apiKey, baseUrl]):
            raise ValueError("模型ID、API密钥和服务地址必须被提供或在.env文件中定义。")

        self.client = OpenAI(api_key=apiKey, base_url=baseUrl, timeout=timeout)

    def think(self, messages: List[Dict[str, str]], temperature: float = 0) -> str:
        """调用 LLM 并以流式方式打印输出，返回完整文本。"""
        print(f"🧠 正在调用 {self.model} 模型...")
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                stream=True,
            )
            # 处理流式响应
            collected_content = []
            for chunk in response:
                if not chunk.choices:
                    continue
                content = chunk.choices[0].delta.content or ""
                print(content, end="", flush=True)
                collected_content.append(content)
            print()  # 在流式输出结束后换行
            print("✅ 大语言模型响应成功:")
            return "".join(collected_content)

        except Exception as e:
            print(f"❌ 调用LLM API时发生错误: {e}")
            return None

    def chat(self, messages: List[Dict[str, str]], tools: List[Dict[str, Any]] = None,
             temperature: float = 0):
        """非流式调用，支持原生 function calling。

        返回 assistant 的 message 对象，调用方可通过 message.tool_calls 判断
        LLM 是否要调用工具。用于让 LLM 自主决定调用工具。
        """
        print(f"🧠 正在调用 {self.model} 模型（function calling）...")
        try:
            kwargs = dict(model=self.model, messages=messages, temperature=temperature)
            if tools:
                kwargs["tools"] = tools
            response = self.client.chat.completions.create(**kwargs)
            print("✅ 大语言模型响应成功。")
            return response.choices[0].message
        except Exception as e:
            print(f"❌ 调用LLM API时发生错误: {e}")
            return None

