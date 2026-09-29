from decimal import Decimal
from functools import lru_cache
from pathlib import Path
import json
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import psycopg
from dotenv import dotenv_values
from FlagEmbedding import BGEM3FlagModel

from app.rag.query_parser import parse_query


ENV_PATH = (
    Path(__file__).resolve().parents[3]
    / "backend"
    / ".env"
)


def get_env():
    return dotenv_values(ENV_PATH)


@lru_cache(maxsize=1)
def get_model():
    return BGEM3FlagModel(
        "BAAI/bge-m3",
        use_fp16=False,
    )


def get_menu_costs(
    authorization: str | None = None,
):
    env = get_env()

    backend_url = env.get(
        "BACKEND_URL",
        "http://localhost:8080",
    ).rstrip("/")

    query_string = urlencode({
        "mealCount": 1,
    })

    url = (
        f"{backend_url}"
        f"/api/cost/menus"
        f"?{query_string}"
    )

    headers = {
        "Accept": "application/json",
    }

    if authorization:
        headers["Authorization"] = authorization

    request = Request(
        url,
        method="GET",
        headers=headers,
    )

    try:
        with urlopen(
            request,
            timeout=15,
        ) as response:
            data = json.loads(
                response.read().decode("utf-8")
            )

    except HTTPError as e:
        if e.code == 401:
            raise RuntimeError(
                "원가 API 인증에 실패했습니다. "
                "로그인 토큰이 전달되지 않았거나 "
                "유효하지 않습니다."
            ) from e

        raise RuntimeError(
            f"원가 API 호출 실패: HTTP {e.code}"
        ) from e

    except URLError as e:
        raise RuntimeError(
            f"원가 API에 연결할 수 없습니다: {url}"
        ) from e

    if isinstance(data, dict):
        data = (
            data.get("data")
            or data.get("content")
            or data.get("items")
            or []
        )

    if not isinstance(data, list):
        raise RuntimeError(
            "원가 API 응답 형식을 확인할 수 없습니다."
        )

    costs = {}

    for item in data:
        menu_id = item.get("menuId")
        cost_per_person = item.get(
            "costPerPerson"
        )

        if (
            menu_id is None
            or cost_per_person is None
        ):
            continue

        costs[int(menu_id)] = Decimal(
            str(cost_per_person)
        )

    return costs


def search_menus(
    query: str,
    authorization: str | None = None,
    conditions: dict | None = None,
):
    if conditions is None:
        conditions = parse_query(query)
    print("[QUERY]", query)
    print("[CONDITIONS]", conditions)

    max_price = conditions.get(
        "max_price"
    )

    category = conditions.get(
        "category"
    )

    price_scope = conditions.get(
        "price_scope"
    )

    slot = conditions.get(
        "slot"
    )

    include_ingredients = (
        conditions.get(
            "include_ingredients"
        )
        or []
    )

    exclude_ingredients = (
        conditions.get(
            "exclude_ingredients"
        )
        or []
    )

    if (
        max_price is not None
        and price_scope == "meal"
    ):
        raise ValueError(
            "식단 전체 예산 조건은 "
            "현재 메뉴 단위 검색에서는 "
            "처리할 수 없습니다."
        )

    allowed_menu_ids = None

    # 가격 조건이 없어도 원가 데이터는 항상 조회
    menu_costs = get_menu_costs(
        authorization=authorization
    )

    if max_price is not None:
        max_price_decimal = Decimal(
            str(max_price)
        )

        allowed_menu_ids = [
            menu_id
            for menu_id, cost
            in menu_costs.items()
            if cost <= max_price_decimal
        ]

        if not allowed_menu_ids:
            return []

    result = get_model().encode(
        [query],
        max_length=128,
        return_dense=True,
        return_sparse=False,
        return_colbert_vecs=False,
    )

    vector = (
        "["
        + ",".join(
            map(
                str,
                result["dense_vecs"][0],
            )
        )
        + "]"
    )

    MAX_DISTANCE = 0.45
    
    sql = """
        SELECT
            m.menu_id,
            r.menu_code,
            m.name,
            m.upper_category,
            m.category,
            m.slot_type,
            m.energy_kcal,
            m.protein_g,
            m.fat_g,
            m.carbohydrate_g,
            m.sodium_mg,
            r.embedding <=> %s::vector
                AS distance
        FROM mealfit.rag_document r
        JOIN mealfit.menu m
            ON r.menu_code = m.menu_code
        WHERE
            r.embedding_model = 'BAAI/bge-m3'
    """

    params = [vector]

    if category:
        sql += """
            AND (
                m.upper_category ILIKE %s
                OR m.category ILIKE %s
            )
        """

        params.extend([
            f"%{category}%",
            f"%{category}%",
        ])

    if slot:
        sql += """
            AND m.slot_type = %s
        """

        params.append(slot)

    for ingredient in include_ingredients:
        sql += """
            AND EXISTS (
                SELECT 1
                FROM mealfit.menu_ingredient mi_inc
                JOIN mealfit.ingredient i_inc
                    ON i_inc.ingredient_id
                     = mi_inc.ingredient_id
                WHERE
                    mi_inc.menu_id = m.menu_id
                    AND i_inc.name ILIKE %s
            )
        """

        params.append(
            f"%{ingredient}%"
        )

    for ingredient in exclude_ingredients:
        sql += """
            AND NOT EXISTS (
                SELECT 1
                FROM mealfit.menu_ingredient mi_exc
                JOIN mealfit.ingredient i_exc
                    ON i_exc.ingredient_id
                     = mi_exc.ingredient_id
                WHERE
                    mi_exc.menu_id = m.menu_id
                    AND i_exc.name ILIKE %s
            )
        """

        params.append(
            f"%{ingredient}%"
        )

    if allowed_menu_ids is not None:
        sql += """
            AND m.menu_id = ANY(%s::bigint[])
        """

        params.append(
            allowed_menu_ids
        )

    sql += """
        ORDER BY distance
        LIMIT 30
    """

    env = get_env()

    with psycopg.connect(
        env["DB_URL"].removeprefix(
            "jdbc:"
        ),
        user=env["DB_USERNAME"],
        password=env["DB_PASSWORD"],
    ) as conn:

        with conn.cursor() as cur:
            cur.execute(
                sql,
                params,
            )

            rows = cur.fetchall()

    results = []

    for (
        menu_id,
        menu_code,
        name,
        upper_category,
        sub_category,
        slot_type,
        energy_kcal,
        protein_g,
        fat_g,
        carbohydrate_g,
        sodium_mg,
        distance,
    ) in rows:

        distance_float = float(distance)
        similarity = (1 - distance_float) * 100

        print(
            f"[RAG] {name} | "
            f"distance={distance_float:.4f} | "
            f"similarity={similarity:.1f}%"
        )

        cost = menu_costs.get(menu_id)

        # 유사도 40% 미만 제외
        if similarity < 40:
            continue

        if similarity < 40:
            continue

        # 가격 조건이 있는 검색에서는
        # 정상 가격 데이터가 반드시 있어야 함
        if max_price is not None:
            if cost is None or cost <= 0:
                continue

        # 가격 조건이 없는 검색에서는
        # 실제 0원 데이터만 제외하고 가격 미등록 메뉴는 허용
        else:
            if cost is not None and cost <= 0:
                continue

        results.append({
        "menu_id": menu_id,
        "menu_code": menu_code,
        "name": name,
        "main_category": upper_category,
        "sub_category": sub_category,
        "slot_type": slot_type,
        "cost_per_person": (
            float(cost)
            if cost is not None
            else None
        ),
        "energy_kcal": (
            float(energy_kcal)
            if energy_kcal is not None
            else None
        ),
        "protein_g": (
            float(protein_g)
            if protein_g is not None
            else None
        ),
        "fat_g": (
            float(fat_g)
            if fat_g is not None
            else None
        ),
        "carbohydrate_g": (
            float(carbohydrate_g)
            if carbohydrate_g is not None
            else None
        ),
        "sodium_mg": (
            float(sodium_mg)
            if sodium_mg is not None
            else None
        ),
        "distance": distance_float,
        "similarity": round(similarity, 1),
    })

    # 유사도 높은 순
    results.sort(
        key=lambda menu: menu["similarity"],
        reverse=True,
    )

    # 최종 3개만 반환
    return results[:30]
