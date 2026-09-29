from typing import Any, TypedDict


class AgentState(TypedDict, total=False):
    query: str
    authorization: str | None

    intent: str

    conditions: dict[str, Any]

    menus: list[dict[str, Any]]

    price_validation: dict[str, Any]
    nutrition_validation: dict[str, Any]
    safety_validation: dict[str, Any]

    answer: str

    error: str