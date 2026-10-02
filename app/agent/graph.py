from langgraph.graph import StateGraph, START, END
from mcp import Client

from app.agent.state import AgentState
from app.agent.nodes import (
    intent_node,
    parse_query_node,
    price_validation_node,
    nutrition_validation_node,
    safety_validation_node,
    recommendation_node,
)

from app.mcp_server import mcp


# MCP를 통해 메뉴 검색 툴 호출
async def search_menu_via_mcp(
    query: str,
    authorization: str | None = None,
    conditions: dict | None = None,
):
    print("[MCP] search_menu 호출")

    async with Client(mcp) as client:
        result = await client.call_tool(
            "search_menu",
            {
                "query": query,
                "authorization": authorization,
                "conditions": conditions,
            },
        )

    print("[MCP] search_menu 결과:", result)

    # MCP 툴 실행 자체가 실패한 경우
    if getattr(result, "is_error", False):
        error_message = "MCP 메뉴 검색 실행에 실패했습니다."

        if getattr(result, "content", None):
            first_content = result.content[0]

            if hasattr(first_content, "text"):
                error_message = first_content.text

        raise RuntimeError(error_message)

    # structured_content에 실제 결과가 있는 경우
    structured_content = getattr(
        result,
        "structured_content",
        None,
    )

    if structured_content is not None:
        if isinstance(structured_content, list):
            return structured_content

        if isinstance(structured_content, dict):
            if "menus" in structured_content:
                return structured_content["menus"]

            if "result" in structured_content:
                return structured_content["result"]

    # data 속성에 실제 결과가 있는 경우
    data = getattr(
        result,
        "data",
        None,
    )

    if isinstance(data, list):
        return data

    if isinstance(data, dict):
        if "menus" in data:
            return data["menus"]

        if "result" in data:
            return data["result"]

    # content에 JSON 문자열이 들어오는 경우
    content = getattr(
        result,
        "content",
        None,
    )

    if content:
        first_content = content[0]

        if hasattr(first_content, "text"):
            import json

            try:
                parsed = json.loads(
                    first_content.text
                )

                if isinstance(parsed, list):
                    return parsed

                if isinstance(parsed, dict):
                    if "menus" in parsed:
                        return parsed["menus"]

                    if "result" in parsed:
                        return parsed["result"]

            except json.JSONDecodeError:
                pass

    return []


# 가격 조회 노드
async def price_lookup_node(state: AgentState) -> dict:
    print("[AGENT] price_lookup_node")

    menus = await search_menu_via_mcp(
        query=state["query"],
        authorization=state.get("authorization"),
        conditions=state.get("conditions"),
    )

    return {
        "menus": menus,
    }


# 식단 분석 노드
async def meal_analysis_node(state: AgentState) -> dict:
    print("[AGENT] meal_analysis_node")

    menus = await search_menu_via_mcp(
        query=state["query"],
        authorization=state.get("authorization"),
        conditions=state.get("conditions"),
    )

    return {
        "menus": menus,
    }


# 사용자 의도에 따라 실행할 노드 결정
def route_by_intent(state: AgentState) -> str:
    intent = state.get("intent")

    if intent == "PRICE_LOOKUP":
        return "PRICE_LOOKUP"

    return "MEAL_ANALYSIS"


# LangGraph 생성
builder = StateGraph(AgentState)

# 사용자 의도 분석
builder.add_node(
    "intent",
    intent_node,
)

# 자연어 질의 조건 구조화
builder.add_node(
    "parse_query",
    parse_query_node,
)

# 가격 조회
builder.add_node(
    "price_lookup",
    price_lookup_node,
)

# 가격 조건 검증
builder.add_node(
    "price_validation",
    price_validation_node,
)

# 식단 및 메뉴 분석
builder.add_node(
    "meal_analysis",
    meal_analysis_node,
)

# 영양 조건 검증
builder.add_node(
    "nutrition_validation",
    nutrition_validation_node,
)

# 알레르기 및 안전 조건 검증
builder.add_node(
    "safety_validation",
    safety_validation_node,
)

# 최종 메뉴 추천 생성
builder.add_node(
    "recommendation",
    recommendation_node,
)


# 그래프 시작 후 사용자 의도 분석
builder.add_edge(
    START,
    "intent",
)

# 의도 분석 후 자연어 조건 구조화
builder.add_edge(
    "intent",
    "parse_query",
)

# 의도에 따라 가격 조회 또는 식단 분석으로 분기
builder.add_conditional_edges(
    "parse_query",
    route_by_intent,
    {
        "PRICE_LOOKUP": "price_lookup",
        "MEAL_ANALYSIS": "meal_analysis",
    },
)

# 가격 조회 후 가격 조건 검증
builder.add_edge(
    "price_lookup",
    "price_validation",
)

# 가격 검증 후 영양 조건 검증
builder.add_edge(
    "price_validation",
    "nutrition_validation",
)

# 식단 분석 후 영양 조건 검증
builder.add_edge(
    "meal_analysis",
    "nutrition_validation",
)

# 영양 검증 후 안전 조건 검증
builder.add_edge(
    "nutrition_validation",
    "safety_validation",
)

# 안전 검증 후 최종 추천 생성
builder.add_edge(
    "safety_validation",
    "recommendation",
)

# 추천 완료 후 그래프 종료
builder.add_edge(
    "recommendation",
    END,
)

# LangGraph 컴파일
menu_agent_graph = builder.compile()


# 메뉴 추천 Agent 실행
async def run_menu_agent(
    query: str,
    authorization: str | None = None,
):
    result = await menu_agent_graph.ainvoke(
        {
            "query": query,
            "authorization": authorization,
        }
    )

    return result