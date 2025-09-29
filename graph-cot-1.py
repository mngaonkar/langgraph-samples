from langgraph.graph import StateGraph, START, END
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel
import os
from dotenv import load_dotenv
from langchain.schema import StrOutputParser
from typing import TypedDict, List

load_dotenv()

llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
class AgentState(TypedDict):
    problem: str
    reasoning_steps: List[str]
    final_answer: str

# Node 1: analyze the problem and break it down into subproblems
analyze_prompt = PromptTemplate(
    input_variables=["problem"],
    template="Analyze the following problem and break it down into smaller subproblems: {problem}"
)

analyze_chain = analyze_prompt | llm | StrOutputParser()

def analyze_problem(state: AgentState) -> str:
    analysis = analyze_chain.invoke({"problem": state["problem"]})
    state["reasoning_steps"].append(f"Analysis: {analysis}")

    return state

# Step 2: perform step-by-step reasoning
reasoning_prompt = PromptTemplate(
    input_variables=["problem", "reasoning_steps"],
    template="Given the problem and analysis:\n\nProblem: {problem}\n\nAnalysis: {reasoning_steps}\n\nProvide a step-by-step reasoning process to solve the problem."
)

def perform_reasoning(state: AgentState) -> str:
    reasoning_chain = reasoning_prompt | llm | StrOutputParser()
    reasoning_steps = "\n".join(state["reasoning_steps"])
    reasoning =  reasoning_chain.invoke({"problem": state["problem"], 
                                         "reasoning_steps": reasoning_steps})

    state["reasoning_steps"].append(f"Reasoning: {reasoning}")
    return state

# Step 3: generate the final answer
final_answer_prompt = PromptTemplate(
    input_variables=["problem", "reasoning_steps"],
    template="Based on the following reasoning steps, provide a final answer to the problem:\n\nProblem: {problem}\n\nReasoning Steps: {reasoning_steps}\n\nFinal Answer:"
)

def generate_final_answer(state: AgentState) -> str:
    final_answer_chain = final_answer_prompt | llm | StrOutputParser()
    reasoning_steps = "\n".join(state["reasoning_steps"])
    final_answer = final_answer_chain.invoke({"problem": state["problem"], 
                                              "reasoning_steps": reasoning_steps})
    state["final_answer"] = final_answer
    return state

workflow = StateGraph(AgentState)

workflow.add_node("analyze_problem", analyze_problem)
workflow.add_node("perform_reasoning", perform_reasoning)
workflow.add_node("generate_final_answer", generate_final_answer)
workflow.add_edge(START, "analyze_problem")
workflow.add_edge("analyze_problem", "perform_reasoning")
workflow.add_edge("perform_reasoning", "generate_final_answer")
workflow.add_edge("generate_final_answer", END)

graph = workflow.compile()

def main():
    initial_state = {
        "problem": "How to colonize mars?",
        "reasoning_steps": [],
        "final_answer": ""
    }

    final_state = graph.invoke(initial_state)
    print("------------------------------------------------")
    print("Final Answer:", final_state["final_answer"])
    print("------------------------------------------------")
    print("Reasoning Steps:")
    for step in final_state["reasoning_steps"]:
        print(step)
    print("------------------------------------------------")

if __name__ == "__main__":
    main()



