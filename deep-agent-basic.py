from deepagents import create_deep_agent
import os
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv


load_dotenv()


@tool
def get_weather(city: str) -> str:
    """Get current weather for a city."""
    weather_by_city = {
        "new york": "72°F and sunny",
        "san francisco": "64°F and foggy",
        "london": "59°F and rainy",
    }
    return weather_by_city.get(city.lower(), "Weather data is unavailable.")


def require_env(name: str) -> None:
    if not os.getenv(name):
        raise ValueError(f"Please set {name} before running this sample.")


def main() -> None:
    require_env("OPENAI_API_KEY")
    model = ChatOpenAI(model="gpt-4o-mini")
    agent = create_deep_agent(
        name="WeatherAssistant",
        system_prompt="You are a helpful weather assistant.",
        tools=[get_weather],
        model=model,
    )
    response = agent.invoke(
        {
            "messages": [
                {"role": "user", "content": "What is the weather in New York today?"}
            ]
        }
    )
    print(response["messages"][-1].content)


if __name__ == "__main__":
    main()
