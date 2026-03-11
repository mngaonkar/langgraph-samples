from deepagents import create_deep_agent
from dotenv import load_dotenv
from tavily import TavilyClient
from langchain_google_genai import ChatGoogleGenerativeAI
import os
import logging

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

load_dotenv()

# Initialize the Tavily client
tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
logger.info("Tavily client initialized successfully.")

# Initialize the Vertex AI model
model = ChatGoogleGenerativeAI(model="gemini-2.5-pro")
logger.info("Model initialized successfully.")

def search_tool(query: str) -> str:
    """Tool to perform an internet search"""
    results = tavily_client.search(query, max_results=3)
    logger.debug(f"Search results: {results}")
    return "\n".join([result["content"] for result in results["results"]]
                     )
agent = create_deep_agent(
    name="ResearchAgent",
    system_prompt="You are a helpful assistant to perform internet searches.",
    tools=[search_tool],
    model=model)

result = agent.invoke({
    "messages": [{"role": "user", "content": "What is quantum computing?"}]
})

logger.info(f"Agent invoked successfully. Result: {result}")
print(result["messages"][-1].content)

