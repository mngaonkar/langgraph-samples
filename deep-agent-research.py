import os
from typing import Literal

from deepagents import create_deep_agent
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from tavily import TavilyClient


load_dotenv()


def internet_search(
    query: str,
    max_results: int = 5,
    topic: Literal["general", "news", "finance"] = "general",
    include_raw_content: bool = False,
):
    """Run an internet search for research tasks."""
    tavily_client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
    return tavily_client.search(
        query=query,
        max_results=max_results,
        topic=topic,
        include_raw_content=include_raw_content,
    )


research_prompt = """You are an expert research assistant.
Gather credible information and respond with:
1) Key findings
2) Risks/limitations
3) A short final summary
"""

def main() -> None:
    model = ChatOpenAI(model="gpt-4o-mini")
    agent = create_deep_agent(
        name="ResearchAssistant",
        system_prompt=research_prompt,
        tools=[internet_search],
        model=model,
    )
    response = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "Explain quantum computing and practical applications.",
                }
            ]
        }
    )
    print(response["messages"][-1].content)


if __name__ == "__main__":
    main()
