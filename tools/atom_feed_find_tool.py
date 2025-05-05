from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Type, Optional
import re
from bs4 import BeautifulSoup

class AtomFeedFindToolInput(BaseModel):
    html_content: str = Field(description="HTML page content to search for Atom feeds")

class AtomFeedFindTool(BaseTool):
    name:str = "atom_feed_find"
    description:str = "Find an Atom feed and return the links."
    args_schema: Type[BaseModel] = AtomFeedFindToolInput

    def _run(self, html_content: str) -> list[str]:
        feed_links = []
        soup = BeautifulSoup(html_content, 'html.parser')
        # Find all <a> tags with "feed" in href
        for link in soup.find_all('a', href=lambda x: x and 'feed' in x.lower()):
            feed_links.append(link.get('href'))
        
        return feed_links

    def _arun(self, html_content: str) -> list[str]:
            raise NotImplementedError("atom_feed_find does not support async")