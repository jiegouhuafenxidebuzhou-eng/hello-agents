from Agent.LLMClient import HelloAgentsLLM, MyAgent
from Agent.tools.terminal_tool import TerminalTool

agent = MyAgent(name="助手", llm=HelloAgentsLLM())
codebase_path = "."
terminal_tool = TerminalTool(workspace=codebase_path, timeout=60)
agent.add_tool(terminal_tool)
response = agent.run("帮我扫描一下file_agent.py文件")
print(response)
