import re
from decimal import Decimal, InvalidOperation

import psycopg

from app.agent.state import AgentState
from app.rag.query_parser import parse_query
from app.rag.search_menus import get_env


def intent_node(state: AgentState) -> dict:
    query = state["query"]

    price_keywords = [
        "가격",
        "원가",
        "비용",
        "예산",
        "얼마",
    ]

    has_price = (
        any(
            keyword in query
            for keyword in price_keywords
        )
        or re.search(
            r"\d[\d,]*\s*원",
            query,
        )
        is not None
    )

    if has_price:
        intent = "PRICE_LOOKUP"
    else:
        intent = "MEAL_ANALYSIS"

    print(f"[AGENT] intent={intent}")

    return {
        "intent": intent,
    }


def parse_query_node(
    state: AgentState,
) -> dict:
    conditions = parse_query(
        state["query"]
    )

    print(
        f"[AGENT] conditions={conditions}"
    )

    return {
        "conditions": conditions,
    }


def price_validation_node(
    state: AgentState,
) -> dict:
    print(
        "[AGENT] price_validation_node"
    )

    menus = (
        state.get("menus")
        or []
    )

    conditions = (
        state.get("conditions")
        or {}
    )

    min_price = conditions.get(
        "min_price"
    )

    max_price = conditions.get(
        "max_price"
    )

    # 가격 조건 자체가 없는 경우
    if (
        min_price is None
        and max_price is None
    ):
        return {
            "menus": menus,
            "price_validation": {
                "checked": False,
                "reason": "가격 조건 없음",
            },
        }

    min_price_decimal = None
    max_price_decimal = None

    if min_price is not None:
        min_price_decimal = Decimal(
            str(min_price)
        )

    if max_price is not None:
        max_price_decimal = Decimal(
            str(max_price)
        )

    valid_menus = []
    invalid_menus = []

    for menu in menus:
        cost = menu.get(
            "cost_per_person"
        )

        # 가격 정보 없는 메뉴 제외
        if cost is None:
            continue

        try:
            cost_decimal = Decimal(
                str(cost)
            )

        except (
            InvalidOperation,
            ValueError,
            TypeError,
        ):
            invalid_menus.append({
                "menu_id": (
                    menu.get("menu_id")
                ),
                "name": (
                    menu.get("name")
                ),
                "reason": (
                    "가격 형식 오류"
                ),
            })
            continue

        # 0원 이하 가격 제외
        if cost_decimal <= 0:
            invalid_menus.append({
                "menu_id": (
                    menu.get("menu_id")
                ),
                "name": (
                    menu.get("name")
                ),
                "reason": (
                    "0원 이하 가격"
                ),
            })
            continue

        # 최소 가격 조건
        if (
            min_price_decimal
            is not None
            and cost_decimal
            < min_price_decimal
        ):
            invalid_menus.append({
                "menu_id": (
                    menu.get("menu_id")
                ),
                "name": (
                    menu.get("name")
                ),
                "reason": (
                    "최소 가격 미만"
                ),
            })
            continue

        # 최대 가격 조건
        if (
            max_price_decimal
            is not None
            and cost_decimal
            > max_price_decimal
        ):
            invalid_menus.append({
                "menu_id": (
                    menu.get("menu_id")
                ),
                "name": (
                    menu.get("name")
                ),
                "reason": (
                    "최대 가격 초과"
                ),
            })
            continue

        valid_menus.append(
            menu
        )

    validation_result = {
        "checked": True,
        "min_price": min_price,
        "max_price": max_price,
        "valid_count": (
            len(valid_menus)
        ),
        "invalid_count": (
            len(invalid_menus)
        ),
        "invalid_menus": (
            invalid_menus
        ),
    }

    print(
        f"[AGENT] price_validation="
        f"{len(valid_menus)} valid / "
        f"{len(invalid_menus)} invalid"
    )

    print(
        "[PRICE VALID MENUS]",
        [
            (
                menu.get("name"),
                menu.get(
                    "cost_per_person"
                ),
            )
            for menu in valid_menus
        ],
    )

    return {
        "menus": valid_menus,
        "price_validation": (
            validation_result
        ),
    }


def nutrition_validation_node(
    state: AgentState,
) -> dict:
    print(
        "[AGENT] nutrition_validation_node"
    )

    menus = (
        state.get("menus")
        or []
    )

    valid_menus = []
    incomplete_menus = []

    nutrition_fields = [
        "energy_kcal",
        "protein_g",
        "fat_g",
        "carbohydrate_g",
        "sodium_mg",
    ]

    for menu in menus:
        missing_fields = [
            field
            for field
            in nutrition_fields
            if menu.get(field)
            is None
        ]

        if missing_fields:
            incomplete_menus.append({
                "menu_id": (
                    menu.get("menu_id")
                ),
                "name": (
                    menu.get("name")
                ),
                "missing_fields": (
                    missing_fields
                ),
            })
            continue

        valid_menus.append(
            menu
        )

    validation_result = {
        "checked": True,
        "valid_count": (
            len(valid_menus)
        ),
        "incomplete_count": (
            len(
                incomplete_menus
            )
        ),
        "incomplete_menus": (
            incomplete_menus
        ),
    }

    print(
        f"[AGENT] nutrition_validation="
        f"{len(valid_menus)} complete / "
        f"{len(incomplete_menus)} incomplete"
    )

    return {
        "menus": valid_menus,
        "nutrition_validation": (
            validation_result
        ),
    }


def get_ingredient_aliases(
    ingredient: str,
) -> list[str]:
    alias_map = {
        "돼지고기": [
            "돼지고기",
            "돼지",
            "돈육",
            "삼겹살",
            "목살",
            "앞다리",
            "뒷다리",
            "돼지갈비",
            "돈갈비",
            "족발",
        ],
        "소고기": [
            "소고기",
            "쇠고기",
            "우육",
        ],
        "닭고기": [
            "닭고기",
            "닭",
            "계육",
        ],
    }

    return alias_map.get(
        ingredient,
        [ingredient],
    )


def safety_validation_node(
    state: AgentState,
) -> dict:
    print(
        "[AGENT] safety_validation_node"
    )

    menus = (
        state.get("menus")
        or []
    )

    conditions = (
        state.get("conditions")
        or {}
    )

    exclude_ingredients = (
        conditions.get(
            "exclude_ingredients"
        )
        or []
    )

    if not menus:
        return {
            "menus": [],
            "safety_validation": {
                "checked": False,
                "reason": (
                    "검증할 메뉴 없음"
                ),
            },
        }

    if not exclude_ingredients:
        return {
            "menus": menus,
            "safety_validation": {
                "checked": False,
                "reason": (
                    "제외 식재료 조건 없음"
                ),
            },
        }

    menu_ids = [
        menu["menu_id"]
        for menu in menus
        if menu.get(
            "menu_id"
        )
        is not None
    ]

    if not menu_ids:
        return {
            "menus": menus,
            "safety_validation": {
                "checked": False,
                "reason": (
                    "메뉴 ID 없음"
                ),
            },
        }

    env = get_env()

    sql = """
        SELECT
            mi.menu_id,
            i.name
        FROM mealfit.menu_ingredient mi
        JOIN mealfit.ingredient i
            ON i.ingredient_id
             = mi.ingredient_id
        WHERE
            mi.menu_id
            = ANY(%s::bigint[])
    """

    with psycopg.connect(
        env["DB_URL"].removeprefix(
            "jdbc:"
        ),
        user=(
            env["DB_USERNAME"]
        ),
        password=(
            env["DB_PASSWORD"]
        ),
    ) as conn:

        with conn.cursor() as cur:
            cur.execute(
                sql,
                [menu_ids],
            )

            rows = (
                cur.fetchall()
            )

    ingredient_map = {}

    for (
        menu_id,
        ingredient_name,
    ) in rows:
        ingredient_map.setdefault(
            menu_id,
            [],
        ).append(
            ingredient_name
        )

    safe_menus = []
    unsafe_menus = []

    for menu in menus:
        menu_id = menu.get(
            "menu_id"
        )

        menu_name = (
            menu.get("name")
            or ""
        ).lower()

        menu_ingredients = (
            ingredient_map.get(
                menu_id,
                [],
            )
        )

        matched_ingredients = []

        for excluded in (
            exclude_ingredients
        ):
            aliases = (
                get_ingredient_aliases(
                    excluded
                )
            )

            matched = False

            for alias in aliases:
                alias_lower = (
                    alias.lower()
                )

                if (
                    alias_lower
                    in menu_name
                ):
                    matched = True
                    break

                for (
                    ingredient_name
                ) in menu_ingredients:
                    ingredient_lower = (
                        ingredient_name
                        .lower()
                    )

                    if (
                        alias_lower
                        in ingredient_lower
                    ):
                        matched = True
                        break

                if matched:
                    break

            if matched:
                matched_ingredients.append(
                    excluded
                )

        if matched_ingredients:
            unsafe_menus.append({
                "menu_id": menu_id,
                "name": (
                    menu.get("name")
                ),
                "matched_ingredients": (
                    matched_ingredients
                ),
            })

        else:
            safe_menus.append(
                menu
            )

    validation_result = {
        "checked": True,
        "safe_count": (
            len(safe_menus)
        ),
        "unsafe_count": (
            len(unsafe_menus)
        ),
        "unsafe_menus": (
            unsafe_menus
        ),
    }

    print(
        f"[AGENT] safety_validation="
        f"{len(safe_menus)} safe / "
        f"{len(unsafe_menus)} unsafe"
    )

    return {
        "menus": safe_menus,
        "safety_validation": (
            validation_result
        ),
    }


def recommendation_node(
    state: AgentState,
) -> dict:
    print(
        "[AGENT] recommendation_node"
    )

    menus = (
        state.get("menus")
        or []
    )

    if not menus:
        return {
            "menus": [],
            "answer": (
                "조건에 맞는 메뉴를 "
                "찾지 못했습니다."
            ),
        }

    recommended_menus = (
        menus[:3]
    )

    menu_names = [
        menu.get(
            "name",
            "메뉴명 없음",
        )
        for menu
        in recommended_menus
    ]

    answer = (
        "조건에 맞는 메뉴를 추천합니다: "
        + ", ".join(
            menu_names
        )
    )

    print(
        f"[AGENT] recommendation="
        f"{menu_names}"
    )

    return {
        "menus": (
            recommended_menus
        ),
        "answer": answer,
    }