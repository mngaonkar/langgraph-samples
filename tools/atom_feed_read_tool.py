from typing import Any, Optional, Type
from langchain.tools import BaseTool
import feedparser
from pydantic import BaseModel, Field

class AtomFeedReadToolInput(BaseModel):
    link: str = Field(description="feed link to read")

class AtomFeedReadTool(BaseTool):
    """Tool to read an Atom feed and return the parsed data."""
    name: str = "atom_feed_read"
    description: str = "Parse an Atom feed and return the parsed data."
    args_schema: Type[BaseModel] = AtomFeedReadToolInput

    def _run(self, link: str) -> Any:
        feed = feedparser.parse(link)
        if feed.bozo:
            raise ValueError(f"Failed to parse feed: {feed.bozo_exception}")
        
        return feed.entries

    def _arun(self, link: str) -> Any:
            raise NotImplementedError("atom_feed_read does not support async")