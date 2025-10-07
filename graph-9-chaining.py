from langgraph.graph import StateGraph, START, END
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from typing import TypedDict

# Create agent state
class AgentState(TypedDict):
    input: str
    output: str

# Create LLM instance and node
llm = ChatOllama(model="smollm2",
                 base_url="http://localhost:11434",
                 verbose=True)

def llm_node_extract(state: AgentState):
    prompt = ChatPromptTemplate.from_template(
    "Extract the technical specifications from the following text: {input}")

    llm_chain = prompt | llm | StrOutputParser()
    response = llm_chain.invoke({"input": state["input"]})
    
    return {"output": response}

def llm_node_transform(state: AgentState):
    prompt = ChatPromptTemplate.from_template(
    "Convert the input text to JSON format: {input}")

    llm_chain = prompt | llm | StrOutputParser()
    response = llm_chain.invoke({"input": state["input"]})
    
    return {"output": response}

builder = StateGraph(AgentState)
builder.add_node(llm_node_extract, "llm_node_extract")
builder.add_node(llm_node_transform, "llm_node_transform")
builder.add_edge(START, "llm_node_extract")
builder.add_edge("llm_node_extract", "llm_node_transform")
builder.add_edge("llm_node_transform", END)

graph = builder.compile()
response = graph.invoke({"input": "The new laptop model features a 3.5 GHz octa-core processor, 16GB of RAM, and a 1TB NVMe SSD."})
print(response)



