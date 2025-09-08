planner_prompt = """
<Task>
You will help plan the steps to implement a LangGraph application based on the user's request. 
</Task>

<Instructions>
1. Reflect on the user's request and the project scope
2. Use the fetch_doc tool to read this llms.txt file, which gives you access to the LangGraph documentation: {llms_txt}
3. [IMPORTANT]: After reading the llms.txt file, ask the user for clarifications to help refine the project scope.
4. Once you have a clear project scope based on the user's feedback, select the most relevant URLs from the llms.txt file to reference in order to implement the project.
5. Then, produce a short summary with two markdown sections: 
- ## Scope: A short description that lays out the scope of the project with up to 5 bullet points
- ## URLs: A list of the {num_urls} relevant URLs to reference in order to implement the project
6. Finally, transfer to the research agent using the transfer_to_researcher_agent tool.
7. Do not implement the solution yourself, handoff to the researcher agent.
</Instructions>
"""

researcher_prompt = """
<Task>
You will implement the solution to the user's request. 
</Task>

<Instructions>
1. First, reflect of the project Scope as provided by the planner agent.
2. Then, use the fetch_doc tool to fetch and read each URL in the list of URLs provided by the planner agent.
3. Reflect on the information in the URLs.
4. Think carefully.
5. Implement the solution to the user's request using the information in the URLs.
6. If you need further clarification or additional sources to implement the solution, then transfer to transfer_to_planner_agent.
</Instructions>

<Checklist> 
Check that your solution satisfies all bullet points in the project Scope.
</Checklist>
"""

from utils import fetch_doc, print_stream
from langchain_openai import ChatOpenAI
from langchain.chat_models import init_chat_model
from langgraph.prebuilt import create_react_agent
from langgraph_swarm import create_handoff_tool, create_swarm
from langchain_google_vertexai import ChatVertexAI
from langgraph.checkpoint.memory import InMemorySaver
from langchain_ollama import ChatOllama

from dotenv import load_dotenv
load_dotenv()

# model1 = init_chat_model(model="gpt-4o-mini", model_provider="openai")
# model2 = ChatVertexAI(model="gemini-2.0-flash", temperature=0)

model1 = ChatOllama(model="qwen3:8b", temperature=0)
model2 = ChatOllama(model="qwen3:8b", temperature=0)

# result = model1.invoke("What is 2+2?")
# print(f"result = {result.content}")

# result = model2.invoke("What is 2+2?")
# print(f"result = {result.content}")

# Handoff tool to transfer between agents
transfer_to_planner_agent = create_handoff_tool(
    agent_name="planner_agent", 
    description="Transfer user to planner agent for clarifying questions related to user's request"
    )

transfer_to_researcher_agent = create_handoff_tool(
    agent_name="researcher_agent",
    description="Transfer to researcher agent to perform research based on the clarified project scope"
    )

llms_txt = "LangGraph:https://langchain-ai.github.io/langgraph/llms.txt"

planner_prompt_formatted = planner_prompt.format(llms_txt=llms_txt, num_urls=3)

planner_agent = create_react_agent(
    model=model1,
    prompt=planner_prompt_formatted,
    tools=[fetch_doc, transfer_to_researcher_agent],
    name="planner_agent",
)

researcher_agent = create_react_agent(
    model=model2,
    prompt=researcher_prompt,
    tools=[fetch_doc, transfer_to_planner_agent],
    name="researcher_agent",
)

checkpointer = InMemorySaver()
agent_swarm = create_swarm(
    agents=[planner_agent, researcher_agent],
    default_active_agent="planner_agent",
)

app = agent_swarm.compile(checkpointer=checkpointer)
config = {"configurable": {"thread_id": "1"}}

print("Enter a user request to create a LangGraph application (q or quit to exit):")
while True:
    user_input = input("USER> ")
    if user_input == "q" or user_input == "quit":
        exit(0)
    else:
        print_stream(
            app.stream(
                {"messages": [{"role": "user", "content": user_input}]},
                config=config,
                subgraphs=True
            )
        )