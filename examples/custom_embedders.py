"""Custom embedders: swap the default local models for anything you like.

The middleware accepts standard LangChain Embeddings objects directly, any
dense embedder exposing dimension and embed(text), and any sparse embedder
exposing embed_document(text) and embed_query(text). This example uses
OpenAI embeddings while keeping the local SPLADE sparse model.

    uv run python examples/custom_embedders.py
"""

from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from langchain_dynamic_tools import DefaultSparseEmbedder, DynamicToolSelectorMiddleware


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


all_tools = [get_weather, query_sql_database, send_email]

# Standard LangChain embeddings are adapted automatically. The index reads
# the dimension from dense_dim (no probing call) or probes it lazily.
dense = OpenAIEmbeddings(model="text-embedding-3-small")

tool_router = DynamicToolSelectorMiddleware(
    tools=all_tools,
    top_k=2,
    dense_embedder=dense,
    dense_dim=1536,
    sparse_embedder=DefaultSparseEmbedder(),  # keep the local SPLADE model
)

agent = create_agent(
    model=ChatOpenAI(model="gpt-4o-mini"),
    tools=all_tools,
    middleware=[tool_router],
)

result = agent.invoke({"messages": [("user", "Email the team about the launch")]})
print(result["messages"][-1].content)
