import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
import logging
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

async def main():
    client = MultiServerMCPClient(
        {
            "weather": {
                "url": "http://localhost:8000/mcp",
                "transport": "streamable_http"
            }
        }
    )

    tools = await client.get_tools()
    logger.info(f"tools = {tools}")

    model = ChatOpenAI(model="gpt-4o-mini")
    agent = create_react_agent(model, tools)
    weather_input = {"messages": [{"role": "user", "content": "whats the weather in New York?"}]}

    response = await agent.ainvoke(weather_input)

    logger.info(f"response = {response["messages"][-1].content}")

asyncio.run(main())