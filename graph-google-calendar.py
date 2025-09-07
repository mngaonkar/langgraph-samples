import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode
from langchain_openai import ChatOpenAI
import logging
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, ToolMessage
from pydantic import BaseModel, Field
from typing import Type, Any
from langchain.tools import BaseTool
from google.oauth2 import service_account
from googleapiclient.discovery import build
import datetime
from langchain.agents import create_react_agent, AgentExecutor
from dateutil import parser

load_dotenv()
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

class FreeBusyInput(BaseModel):
    start_time: str = Field(..., description="Start time in ISO 8601 format in local timezone")
    end_time: str = Field(..., description="End time in ISO 8601 format in local timezone")
    duration_minutes: int = Field(..., description="Duration in minutes")
    calendar_id: str = Field(..., description="Calendar ID to check availability")

class FreeBusyTool(BaseTool):
    name: str = "free_busy_tool"
    description: str = "Query Google calendar for available time slots within a specified start time in local timezone, end time in local timezone and duration"
    args_schema: Type[BaseModel] = FreeBusyInput

    service_account_file: str = "/Users/mahadevgaonkar/Downloads/trinity-ai-425119-6a11cd0e3584.json"

    @property
    def credentials(self):
        SCOPES = ['https://www.googleapis.com/auth/calendar']
        return service_account.Credentials.from_service_account_file(
            self.service_account_file, scopes=SCOPES)

    @property
    def service(self):
        return build('calendar', 'v3', credentials=self.credentials)


    def _run(self, start_time: str, end_time: str, duration_minutes: int, calendar_id: str = "primary") -> str:
        """Query Google Calendar API for free/busy information."""
        start_time = parser.parse(start_time).isoformat()
        end_time = parser.parse(end_time).isoformat()
        start_time = start_time + '-07:00'  # 'PDT' indicates Pacific Daylight Time
        end_time = end_time + '-07:00'

        logger.debug(f"####### Querying free/busy from {start_time} to {end_time} for duration {duration_minutes} minutes on calendar {calendar_id}")
        try:
            # calendar_id = 'mahadev.gaonkar@gmail.com'  # replace with actual calendar ID

            # self.service.calendarList().insert(
            #     body={
            #         'id': calendar_id
            #     }
            # ).execute()

            # Call the CalendarList API to get calendars
            calendar_list = self.service.calendarList().list().execute()
            logger.debug(f"##### Fetched {len(calendar_list.get('items', []))} calendars")

            # Extract and print calendar IDs and summaries
            for calendar_entry in calendar_list.get('items', []):
                logger.info(f"Calendar Summary: {calendar_entry.get('summary')}")
                logger.info(f"Calendar ID: {calendar_entry.get('id')}")
                logger.info('---')

            # Fetch events
            events_result = self.service.events().list(
                calendarId=calendar_id,
                timeMin=start_time,
                timeMax=end_time,
                singleEvents=True,
                orderBy='startTime'
            ).execute()
            events = events_result.get('items', [])
        except Exception as e:
            return f"Error querying free/busy information: {e}"

        logger.debug(f"##### Fetched {len(events)} events")
        available_slots = []
        start = parser.parse(start_time)
        end = parser.parse(end_time)
        current_time = start

        while current_time + datetime.timedelta(minutes=duration_minutes) <= end:
            slot_end = current_time + datetime.timedelta(minutes=duration_minutes)
            if not any(busy['start'] < slot_end.isoformat() and busy['end'] > current_time.isoformat() for busy in events):
                available_slots.append((current_time.isoformat(), slot_end.isoformat()))
            current_time += datetime.timedelta(minutes=15)

        if not available_slots:
            return "No available slots found."

        result = "Available slots:\n" + "\n".join([f"{slot[0]} to {slot[1]}" for slot in available_slots])
        return result
    
async def main():
    model = ChatOpenAI(model="gpt-4o-mini")
    tools = [FreeBusyTool()]

    model_with_tools = model.bind_tools(tools)
    from langchain.prompts import PromptTemplate

    prompt = PromptTemplate(
        input_variables=["input", "agent_scratchpad", "tools", "tool_names"],
        template=(
            "You are an AI assistant that helps schedule meetings using the provided tools.\n"
            "Available tools: {tools}\n"
            "Tool names: {tool_names}\n"
            "{input}\n"
            "{agent_scratchpad}"
        )
    )
    
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
    query = "Find me a 45-minute free slot on Sep 11, 2025 from  12 AM to 11 PM."
    response = await graph.ainvoke(
        {
            "messages": [{"role": "user", "content": query}]
        }
    )
    logger.info(f"response = {response["messages"][-1].content}")

asyncio.run(main())

