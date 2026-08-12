from Agent.LLMClient import MyAgent, HelloAgentsLLM
from Agent.MCP import MCPTool

agent = MyAgent(name="助手", llm=HelloAgentsLLM())

# 示例1：连接到社区提供的文件系统服务器
fs_tool = MCPTool(
    name="filesystem",  # 指定唯一名称
    description="访问本地文件系统",
    server_command=["npx", "-y", "@modelcontextprotocol/server-filesystem", "."]
)
agent.add_tool(fs_tool)


print("\n当前Agent拥有的工具：")
print(f"- {fs_tool.name}: {fs_tool.description}")
# Agent现在可以自动使用这些工具！
response = agent.run("请读取Sonnar扫描.md文件，并总结其中的主要内容")
print(response)