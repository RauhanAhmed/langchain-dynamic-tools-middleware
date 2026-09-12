"""Quickstart: an agent with many tools that only sees the relevant few.

Set OPENAI_API_KEY before running:

    uv run python examples/quickstart.py
"""

from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from langchain_dynamic_tools import DynamicToolSelectorMiddleware


@tool
def get_weather(city: str) -> str:
    """Get the current weather conditions and forecast for a city."""
    return f"Sunny, 22C in {city}"


@tool
def query_sql_database(sql_query: str) -> str:
    """Run a SQL query against the sales database and return matching rows."""
    return "3 rows returned"


@tool
def send_email(recipient: str, subject: str, body: str) -> str:
    """Send an email message to a recipient with a subject and a body."""
    return f"Email sent to {recipient}"


@tool
def git_push(branch: str, commit_message: str) -> str:
    """Push commits on a branch to the git repository with a commit message."""
    return f"Pushed to {branch}"


@tool
def create_calendar_event(title: str, start_time: str) -> str:
    """Create a calendar event with a title and a start time."""
    return f"Event {title!r} created"


@tool
def search_files(directory: str, pattern: str) -> str:
    """Search for files in a directory matching a name pattern."""
    return "2 matches"


# 1. All the tools your application offers
all_tools = [
    get_weather,
    query_sql_database,
    send_email,
    git_push,
    create_calendar_event,
    search_files,
]

# 2. Add the dynamic middleware. It indexes every tool once with local
#    dense and sparse embeddings, then picks the best matches per step.
tool_router = DynamicToolSelectorMiddleware(
    tools=all_tools,
    top_k=2,
)

# 3. Create the agent. The full set stays registered for execution, but each
#    model call only carries the tools relevant to the user's message.
agent = create_agent(
    model=ChatOpenAI(model="gpt-4o-mini"),
    tools=all_tools,
    middleware=[tool_router],
)

result = agent.invoke({"messages": [("user", "What is the weather in Tokyo right now?")]})
print(result["messages"][-1].content)

result = agent.invoke({"messages": [("user", "Push my work to the release branch")]})
print(result["messages"][-1].content)
