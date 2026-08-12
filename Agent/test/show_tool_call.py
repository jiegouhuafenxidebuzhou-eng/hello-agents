"""实调演示：打印 LLM 原生 function calling 返回的结构化对象。

挂一个 add 工具，问 "计算 123 + 456"，把 assistant 对象原样打印出来，
直观看到 tool_calls 是结构化字段、而不是需要解析的文本。
"""
from Agent.LLMClient import HelloAgentsLLM, FunctionTool


def add(x, y):
    return x + y


def main():
    llm = HelloAgentsLLM()

    # 一个加法工具
    add_tool = FunctionTool(
        name="add",
        description="对两个数字做加法",
        func=add,
        parameters={
            "type": "object",
            "properties": {
                "x": {"type": "number", "description": "第一个加数"},
                "y": {"type": "number", "description": "第二个加数"},
            },
            "required": ["x", "y"],
        },
    )

    messages = [
        {"role": "system", "content": "你是一个能调用工具的助手。"},
        {"role": "user", "content": "请用工具计算 123 + 456"},
    ]

    # 调 chat，拿到 assistant 的 message 对象
    assistant = llm.chat(messages=messages, tools=[add_tool.spec()])

    print("\n" + "=" * 60)
    print("1) assistant 的类型：")
    print("   ", type(assistant))

    print("\n2) assistant.content：")
    print("   ", repr(assistant.content))

    print("\n3) assistant.tool_calls：")
    print("   ", assistant.tool_calls)

    if assistant.tool_calls:
        tc = assistant.tool_calls[0]
        print("\n4) 第一个 tool_call 的拆解：")
        print("   tc.id            =", tc.id)
        print("   tc.type          =", tc.type)
        print("   tc.function.name =", tc.function.name)
        print("   tc.function.arguments =", tc.function.arguments)
        print("   (type of arguments) =", type(tc.function.arguments))

        import json
        args = json.loads(tc.function.arguments)
        print("\n5) json.loads(arguments) 得到的字典：")
        print("   ", args, "->", add(**args))

    print("\n" + "=" * 60)
    print("结论：name / arguments 已经是分好的字段，MyAgent 直接点属性、")
    print("json.loads 一下就能执行，全程不用正则解析文本。")


if __name__ == "__main__":
    main()
