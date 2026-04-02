"""
main.py — Entry point for the LangGraph Planner-Executor agent

Flow:
  START → planner_node → executor_node → (loop until all steps done) → END
"""

import asyncio
import os
import sys

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_mcp_adapters.client import MultiServerMCPClient
from graph import AgentState, build_graph

load_dotenv()

# ─── LLM 
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0, api_key=os.environ["GROQ_API_KEY"])


# ─── MCP client 
mcp = MultiServerMCPClient({
    "math": {
        "command": sys.executable,
        "args": ["Tools/math_server.py"],
        "transport": "stdio",
    },
    "search": {
        "command": sys.executable,
        "args": ["Tools/search_server.py"],
        "transport": "stdio",
    },
    "weather": {
        "url": "http://localhost:8000/mcp",
        "transport": "streamable_http",
    },
})


# ─── Runner 
async def run(goal: str) -> list:
    # Load all MCP tools once before building the graph
    tools = []
    for server in ["math", "search", "weather"]:
        try:
            t = await mcp.get_tools(server_name=server)
            tools.extend(t)
        except Exception as e:
            print(f"[Warning] Could not load '{server}' tools: {e}")

    tools_map = {t.name: t for t in tools}
    print(f"[Setup] Loaded tools: {list(tools_map.keys())}\n")

    # Build and run the graph
    graph = build_graph(llm, tools_map)

    initial_state: AgentState = {
        "goal":         goal,
        "plan":         [],
        "current_step": 0,
        "results":      [],
    }

    final_state = await graph.ainvoke(initial_state)

    # ── Print summary 
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    for r in final_state["results"]:
        print(f"\nStep {r['step']}: {r['description']}")
        print(f"  {r['result'][:400]}")

    return final_state["results"]


# ─── Entry point 
if __name__ == "__main__":
    goal = (
        "Plan an outdoor event for 150 people in Lahore: "
        "calculate tables/chairs (10 people per table), find average ticket price, "
        "check weather in Lahore, and summarize."
    )
    asyncio.run(run(goal))