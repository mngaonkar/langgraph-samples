import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode
from langchain_openai import ChatOpenAI
import logging
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, ToolMessage

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
    model_with_tools = model.bind_tools(tools)

    async def call_model(state: MessagesState):
        response = await model_with_tools.ainvoke(state["messages"])
        logger.debug(f"Model response: {response}")
        return {"messages": [response]}
    
    def should_continue(state: MessagesState):
        last_message = state["messages"][-1]
        logger.debug(f"State messages: {state['messages']}")
        logger.debug(f"Last message: {last_message}")

        if isinstance(last_message, HumanMessage):
            return "tools"
        
        if not last_message.tool_calls:
            return END
        
        return "tools"
    
    tool_node = ToolNode(tools)

    builder = StateGraph(state_schema=MessagesState)
    builder.add_node("call_model", call_model)  
    builder.add_node("tools", tool_node)
    builder.add_edge(START, "call_model")
    builder.add_conditional_edges("call_model", should_continue)
    builder.add_edge("tools", "call_model")

    graph = builder.compile()

    response = await graph.ainvoke(
        {
            "messages": [{"role": "user", "content": "whats the weather in New York?"}]
        }
    )
    logger.info(f"response = {response['messages'][-1].content}")


asyncio.run(main())