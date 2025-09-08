from langchain_tavily import TavilySearch
from langgraph.prebuilt import create_react_agent
from langgraph_supervisor import create_supervisor
from langchain.chat_models import init_chat_model

from dotenv import load_dotenv
load_dotenv()
import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

web_search = TavilySearch(max_results=3)

research_agent = create_react_agent(
    model="openai:gpt-4o",  # Use GPT-4o or similar
    tools=[web_search],
    prompt="""You are a research agent.

INSTRUCTIONS:
- Assist ONLY with research-related tasks, DO NOT do any math
- After you're done with your tasks, respond to the supervisor directly
- Respond ONLY with the results of your work, do NOT include ANY other text.""",
    name="research_agent",
)

def add(a: float, b: float) -> float:
    """Add two numbers."""
    return a + b

def multiply(a: float, b: float) -> float:
    """Multiply two numbers."""
    return a * b

def divide(a: float, b: float) -> float:
    """Divide two numbers."""
    return a / b

math_agent = create_react_agent(
    model="openai:gpt-4o",
    tools=[add, multiply, divide],
    prompt="""You are a math agent.

INSTRUCTIONS:
- Assist ONLY with math-related tasks
- After you're done with your tasks, respond to the supervisor directly
- Respond ONLY with the results of your work, do NOT include ANY other text.""",
    name="math_agent",
)

supervisor = create_supervisor(
    model=init_chat_model("openai:gpt-4o"),
    agents=[research_agent, math_agent],
    prompt="""You are a supervisor managing two agents:
- a research agent. Assign research-related tasks to this agent
- a math agent. Assign math-related tasks to this agent

Assign work to one agent at a time, do not call agents in parallel.
Do not do any work yourself.""",
    add_handoff_back_messages=True,  # Includes previous messages when handing back
    output_mode="full_history",  # Returns full conversation history
).compile()

config = {"configurable": {"thread_id": "supervisor-thread"}}
query = "What is the total and individual market cap of top 10 companies in the US?"

for event in supervisor.stream(
    {"messages": [("user", query)]},
    config,
    stream_mode="values",
):
    event["messages"][-1].pretty_print()