from deepagents import create_deep_agent
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
import os

load_dotenv()

# Initialize the Vertex AI model
model = ChatGoogleGenerativeAI(model="gemini-2.5-pro")
print("Model initialized successfully.")

def get_weather(city: str) -> str:
    """Tool to get weather given a city"""
    return "The weather is sunny."

agent = create_deep_agent(
    name="WeatherAgent",
    system_prompt="You are a helpful assistant to get the weather.",
    tools=[get_weather],
    model=model)

result = agent.invoke({
    "messages": [{"role": "user", "content": "What is the weather in New York?"}]
})

print(f"Agent invoked successfully. Result: {result}")
print(result["messages"][-1].content)

