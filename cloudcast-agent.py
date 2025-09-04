# Import relevant functionality
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from langchain_ollama import ChatOllama
from tools.visit_web_page_tool import VisitWebPageSyncTool
from functions.atom_feed_find_func import AtomFeedFindTool
from functions.atom_feed_read_func import AtomFeedReadTool
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.tools.retriever import create_retriever_tool
from langchain_openai import ChatOpenAI 
import os, json
from langchain_core.documents import Document
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from dotenv import load_dotenv
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

load_dotenv()

DOC_LOCATION = "/Users/mahadevgaonkar/code/hf-agent-course/tools/transcripts/"
MAX_RETRIEVAL = 3

llm = ChatOpenAI(
    temperature=0,
    model="gpt-4o")

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Create a text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100, 
    chunk_overlap=20,
    separators=["\n\n", "\n", " ", ""])

# Load text files from directory and split them into chunks
documents = []
for file in os.listdir(DOC_LOCATION):
    with open(os.path.join(DOC_LOCATION, file), 'r') as f:
        text = f.read()
        doc = text_splitter.split_text(text)
        documents.append({
            "text": doc,
            "metadata": {"source": os.path.join(DOC_LOCATION, file),
                         "link": ""}
        })

doc_list = [Document(page_content=json.dumps(doc["text"]), metadata=doc["metadata"]) for doc in documents]

# Initialize the vector store with the documents
vector_store = Chroma.from_documents(
    documents=doc_list,
    embedding=embeddings,
    persist_directory="chroma_db"
)

# Define state
class GraphState(TypedDict):
    query: str
    web_search_result: str
    results: list[Document]

def query_vector_store(state: GraphState) -> GraphState:
    """Function to query the vector store."""
    result = vector_store.similarity_search(state["query"], k=MAX_RETRIEVAL)
    state["results"] = result
    return state

def print_search_results(state: GraphState) -> None:
    """Function to print search results."""
    for doc in state["results"]:
        print(f"Content: {doc.page_content[:100]}..., Source: {doc.metadata['source']}")  # Print first 100 characters of content

llm_with_tools = llm.bind_tools(
    [VisitWebPageSyncTool(),
     AtomFeedFindTool(),
     AtomFeedReadTool()]
)

def tool_calling_llm(state: GraphState) -> GraphState:
    """Function to call the LLM with tools."""
    response = llm_with_tools.invoke(
        [HumanMessage(content=state["query"])]
    )
    print(f"LLM Response: {response}")
    state["web_search_result"] = response.content
    return state

# Create a state graph
builder = StateGraph(GraphState)
builder.add_node("query_database", query_vector_store)
builder.add_node("print_results", print_search_results)
builder.add_node("tool_calling_llm", tool_calling_llm)

builder.add_edge(START, "tool_calling_llm")
builder.add_edge("tool_calling_llm", "query_database")
builder.add_edge("query_database", "print_results")
builder.add_edge("print_results", END)

graph = builder.compile()

# Invoke the graph with a query
result = graph.invoke(
    GraphState(query="find atom feed in https://www.thecloudcast.net")
)

logger.info(f"Final result: {result}")