from langgraph.graph import StateGraph, START, END

from app.agent.state import AgentState
from app.agent.nodes import (
    intent_node,
    parse_query_node,
    price_validation_node,
    nutrition_validation_node,
    safety_validation_node,
    recommendation_node,
)
from app.rag.search_menus import search_menus


def price_lookup_node(state: AgentState) -> dict:
    print("[AGENT] price_lookup_node")

    menus = search_menus(
        state["query"],
        authorization=state.get("authorization"),
        conditions=state.get("conditions"),
    )

    return {
        "menus": menus,
    }


def meal_analysis_node(state: AgentState) -> dict:
    print("[AGENT] meal_analysis_node")

    menus = search_menus(
        state["query"],
        authorization=state.get("authorization"),
        conditions=state.get("conditions"),
    )

    return {
        "menus": menus,
    }


def route_by_intent(state: AgentState) -> str:
    intent = state.get("intent")

    if intent == "PRICE_LOOKUP":
        return "PRICE_LOOKUP"

    return "MEAL_ANALYSIS"


builder = StateGraph(AgentState)

builder.add_node(
    "intent",
    intent_node,
)

builder.add_node(
    "parse_query",
    parse_query_node,
)

builder.add_node(
    "price_lookup",
    price_lookup_node,
)

builder.add_node(
    "price_validation",
    price_validation_node,
)

builder.add_node(
    "meal_analysis",
    meal_analysis_node,
)

builder.add_node(
    "nutrition_validation",
    nutrition_validation_node,
)

builder.add_node(
    "safety_validation",
    safety_validation_node,
)

builder.add_node(
    "recommendation",
    recommendation_node,
)

builder.add_edge(
    START,
    "intent",
)

builder.add_edge(
    "intent",
    "parse_query",
)

builder.add_conditional_edges(
    "parse_query",
    route_by_intent,
    {
        "PRICE_LOOKUP": "price_lookup",
        "MEAL_ANALYSIS": "meal_analysis",
    },
)

# 가격 조회
builder.add_edge(
    "price_lookup",
    "price_validation",
)

builder.add_edge(
    "price_validation",
    "nutrition_validation",
)

# 식단 분석
builder.add_edge(
    "meal_analysis",
    "nutrition_validation",
)

# 영양 검증 후 안전 검증
builder.add_edge(
    "nutrition_validation",
    "safety_validation",
)

# 안전 검증 후 추천
builder.add_edge(
    "safety_validation",
    "recommendation",
)

builder.add_edge(
    "recommendation",
    END,
)

menu_agent_graph = builder.compile()


def run_menu_agent(
    query: str,
    authorization: str | None = None,
):
    result = menu_agent_graph.invoke({
        "query": query,
        "authorization": authorization,
    })

    return result