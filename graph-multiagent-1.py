import json
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    BaseMessage,
    ChatMessage,
    FunctionMessage,
    SystemMessage
)
from typing import Literal, Optional, Union, List, Dict, Any
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode
from langchain_openai import ChatOpenAI
from langgraph.prebuilt.chat_agent_executor import create_react_agent
from dotenv import load_dotenv

load_dotenv()
import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

model = ChatOpenAI(model="gpt-4o-mini")

def agent_1(state: MessagesState):
    system_message = """You are agent 1 specialized in math calculations. You will do the first level of reasoning and pass the output to agent 2. If you sure about the answer, please add FINAL ANSWER keyword in output."""
    messages = [SystemMessage(content=system_message)] + state["messages"]
    response = model.invoke(messages)
    return {"messages": [response]}

def agent_2(state: MessagesState):
    response = model.invoke(state["messages"])
    return {"messages": [response]}

def router(state: MessagesState) -> Literal["agent_2", END]:
    last_message = state["messages"][-1]
    if "FINAL ANSWER" in last_message.content:
        return END

    return "agent_2"

builder = StateGraph(state_schema=MessagesState)
builder.add_node("agent_1", agent_1)
builder.add_node("agent_2", agent_2)
builder.add_edge(START, "agent_1")
builder.add_conditional_edges("agent_1", router)
builder.add_edge("agent_2", "agent_1")

graph = builder.compile()

initial_state = {"messages": [HumanMessage(content="What is 2+2. Explain step by step?")]}

final_state = graph.invoke(initial_state)
# logger.info(f"result = {final_state['messages'][-1].content}")
for message in final_state['messages']:
    message.pretty_print()