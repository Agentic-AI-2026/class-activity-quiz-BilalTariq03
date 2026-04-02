"""
graph.py — LangGraph Planner-Executor: state definition + graph builder
"""

import json
import re
from typing import TypedDict

from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, SystemMessage


# ─── 1. State 

class AgentState(TypedDict):
    goal:         str    # user input
    plan:         list   # list of step dicts produced by the planner
    current_step: int    # index of the step being executed
    results:      list   # outputs accumulated from executed steps


# ─── 2. Prompts / helpers 

PLAN_SYSTEM = """Break the user goal into an ordered JSON list of steps.
Each step MUST follow this EXACT schema:
  {"step": int, "description": str, "tool": str or null, "args": dict or null}

Available tools and their EXACT argument names:
  - calculator(expression: str)         → evaluate a math expression safely
  - multiply(a: float, b: float)        → multiply two numbers
  - divide(a: float, b: float)          → divide a by b
  - search_web(query: str)              → search the web for information
  - search_news(query: str)             → search for recent news
  - get_current_weather(city: str)      → get real-time weather for a city

Use null for tool/args on synthesis or summary steps.
Return ONLY a valid JSON array — no markdown fences, no explanation."""

# Maps tool name, the single positional arg name (for remapping hallucinated keys)
TOOL_ARG_MAP = {
    "calculator":          "expression",
    "search_web":          "query",
    "search_news":         "query",
    "get_current_weather": "city",
    "get_weather_forecast":"city",
}


def safe_args(tool_name: str, raw_args: dict) -> dict:
    """Remap hallucinated argument names to the correct parameter."""
    expected = TOOL_ARG_MAP.get(tool_name)
    if not expected or expected in raw_args:
        return raw_args
    first_val = next(iter(raw_args.values()), tool_name)
    print(f"  [safe_args] Remapped {raw_args} → {{'{expected}': '{first_val}'}}")
    return {expected: str(first_val)}


# ─── 3. Graph builder (receives pre-loaded tools_map + llm) 

def build_graph(llm, tools_map: dict):
    """
    Builds and compiles the LangGraph workflow.

    Flow:  START → planner_node → executor_node ──(loop)──→ END
    """

    # ── Planner Node 
    async def planner_node(state: AgentState) -> dict:
        print(f"\n[Planner] Goal: {state['goal']}\n")

        resp = llm.invoke([
            SystemMessage(content=PLAN_SYSTEM),
            HumanMessage(content=state["goal"]),
        ])

        raw = resp.content if isinstance(resp.content, str) else resp.content[0].get("text", "")
        clean = re.sub(r"```json|```", "", raw).strip()
        plan = json.loads(clean)

        print(f"[Planner] Generated {len(plan)} steps:")
        for s in plan:
            print(f"  Step {s['step']}: {s['description']}  |  tool={s.get('tool')}")

        return {"plan": plan, "current_step": 0, "results": []}

    # ── Executor Node 
    async def executor_node(state: AgentState) -> dict:
        idx  = state["current_step"]
        step = state["plan"][idx]

        print(f"\n[Executor] Step {step['step']}: {step['description']}")

        tool_name = step.get("tool")

        if tool_name and tool_name in tools_map:
            corrected = safe_args(tool_name, step.get("args") or {})
            raw_result = await tools_map[tool_name].ainvoke(corrected)
            # Extract plain text if result is a list of content blocks
            if isinstance(raw_result, list):
                texts = [
                    block.get("text", "").strip() if isinstance(block, dict) else str(block).strip()
                    for block in raw_result
                ]
                result = next((t for t in texts if t), str(raw_result))
            else:
                result = raw_result
        else:
            # Synthesis / summary step give the LLM all prior results as context
            context  = "\n".join(
                f"Step {r['step']} ({r['description']}): {r['result']}" for r in state["results"]
            )
            resp   = llm.invoke([
                HumanMessage(content=(
                    f"Goal: {state['goal']}\n\n"
                    f"Results so far:\n{context}\n\n"
                    f"Now complete this step: {step['description']}"
                ))
            ])
            result = resp.content

        result_str = str(result)
        print(f"  → {result_str[:250]}")

        new_results = state["results"] + [{
            "step":        step["step"],
            "description": step["description"],
            "result":      result_str,
        }]
        return {"current_step": idx + 1, "results": new_results}

    # ── Routing condition 
    def should_continue(state: AgentState) -> str:
        if state["current_step"] >= len(state["plan"]):
            return "end"
        return "continue"

    # ── Assemble graph 
    workflow = StateGraph(AgentState)

    workflow.add_node("planner_node",  planner_node)
    workflow.add_node("executor_node", executor_node)

    workflow.add_edge(START, "planner_node")
    workflow.add_edge("planner_node", "executor_node")
    workflow.add_conditional_edges(
        "executor_node",
        should_continue,
        {"continue": "executor_node", "end": END},
    )

    return workflow.compile()