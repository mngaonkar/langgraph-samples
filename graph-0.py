# Import relevant functionality
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from langchain_ollama import ChatOllama

# Create the agent
memory = MemorySaver()

# Create Ollama model
model = ChatOllama(
    model = "qwen2.5-coder",
    temperature = 0.8,
    num_predict = 256,
    base_url = "http://10.0.0.147:11434"
)

agent_executor = create_react_agent(
    model=model,
    tools=[],
    checkpointer=memory)

# Use the agent
config = {"configurable": {"thread_id": "abc123"}}
for step in agent_executor.stream(
    {"messages": [HumanMessage(content="hi im bob! and i live in sf")]},
    config,
    stream_mode="values",
):
    step["messages"][-1].pretty_print()

