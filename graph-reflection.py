from typing import TypedDict, Annotated
from operator import add

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

# Define the state schema
class State(TypedDict):
    task: str
    output: str
    critique: str
    needs_refinement: bool
    iteration: Annotated[int, add]

# Dummy functions (replace with actual agent logic, e.g., LLM calls)
def generate_initial(state: State) -> dict:
    # Generate initial output based on task
    initial_output = f"Initial output for task: {state['task']}"
    return {"output": initial_output}

def critique(state: State) -> dict:
    # Critique the current output
    crit = f"Critique of output: {state['output']} (iteration {state['iteration']})"
    # Dummy condition: needs refinement for first 2 iterations
    needs = state['iteration'] < 2
    return {"critique": crit, "needs_refinement": needs}

def refine(state: State) -> dict:
    # Refine the output based on critique
    refined_output = f"Refined output: {state['output']} after critique: {state['critique']}"
    return {"output": refined_output, "iteration": 1}  # Increment iteration

# Conditional function for routing
def router(state: State):
    if state["needs_refinement"]:
        return "refine"
    else:
        return END

# Build the graph
builder = StateGraph(State)

# Add nodes
builder.add_node("generate_initial", generate_initial)
builder.add_node("critique", critique)
builder.add_node("refine", refine)

# Add edges
builder.add_edge(START, "generate_initial")
builder.add_edge("generate_initial", "critique")

# Conditional edge after critique
builder.add_conditional_edges("critique", router, {"refine": "refine", END: END})

# After refine, go back to critique
builder.add_edge("refine", "critique")

# Compile the graph with memory for state persistence
memory = MemorySaver()
graph = builder.compile(checkpointer=memory)

# Example usage
if __name__ == "__main__":
    initial_state = {"task": "Sample task", "output": "", "critique": "", "needs_refinement": True, "iteration": 0}
    config = {"configurable": {"thread_id": "reflection_thread"}}
    result = graph.invoke(initial_state, config)
    print(result)
    # The output will show the final state after iterations