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


ENV_PATH = Path(__file__).resolve().parents[2] / ".env"


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
                response.read().decode(
                    "utf-8"
                )
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

    if isinstance(
        data,
        dict,
    ):
        data = (
            data.get("data")
            or data.get("content")
            or data.get("items")
            or []
        )

    if not isinstance(
        data,
        list,
    ):
        raise RuntimeError(
            "원가 API 응답 형식을 확인할 수 없습니다."
        )

    costs = {}

    for item in data:
        menu_id = item.get(
            "menuId"
        )

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


def find_weight_column(conn):
    candidates = [
        "weight",
        "weight_g",
        "serving_weight_g",
        "standard_weight_g",
    ]

    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT column_name
            FROM information_schema.columns
            WHERE table_schema = 'mealfit'
              AND table_name = 'menu'
            """
        )

        existing_columns = {
            row[0]
            for row in cur.fetchall()
        }

    for candidate in candidates:
        if candidate in existing_columns:
            return candidate

    return None


def search_menus(
    query: str,
    authorization: str | None = None,
    conditions: dict | None = None,
):
    if conditions is None:
        conditions = parse_query(
            query
        )

    print(
        "[QUERY]",
        query,
    )

    print(
        "[CONDITIONS]",
        conditions,
    )

    min_price = conditions.get(
        "min_price"
    )

    max_price = conditions.get(
        "max_price"
    )

    category = conditions.get(
        "category"
    )

    menu_keyword = conditions.get(
        "menu_keyword"
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

    min_weight = conditions.get(
        "min_weight"
    )

    max_weight = conditions.get(
        "max_weight"
    )

    min_calories = conditions.get(
        "min_calories"
    )

    max_calories = conditions.get(
        "max_calories"
    )

    min_protein = conditions.get(
        "min_protein"
    )

    max_protein = conditions.get(
        "max_protein"
    )

    min_sodium = conditions.get(
        "min_sodium"
    )

    max_sodium = conditions.get(
        "max_sodium"
    )

    if (
        (
            min_price is not None
            or max_price is not None
        )
        and price_scope == "meal"
    ):
        raise ValueError(
            "식단 전체 예산 조건은 "
            "현재 메뉴 단위 검색에서는 "
            "처리할 수 없습니다."
        )

    menu_costs = get_menu_costs(
        authorization=authorization
    )

    allowed_menu_ids = None

    if (
        min_price is not None
        or max_price is not None
    ):
        min_price_decimal = (
            Decimal(
                str(min_price)
            )
            if min_price is not None
            else None
        )

        max_price_decimal = (
            Decimal(
                str(max_price)
            )
            if max_price is not None
            else None
        )

        allowed_menu_ids = []

        for (
            menu_id,
            cost,
        ) in menu_costs.items():

            if (
                min_price_decimal is not None
                and cost < min_price_decimal
            ):
                continue

            if (
                max_price_decimal is not None
                and cost > max_price_decimal
            ):
                continue

            allowed_menu_ids.append(
                menu_id
            )

        if not allowed_menu_ids:
            return []

    # 사용자 질문을 임베딩
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
                result[
                    "dense_vecs"
                ][0],
            )
        )
        + "]"
    )

    env = get_env()

    with psycopg.connect(
        env[
            "DB_URL"
        ].removeprefix(
            "jdbc:"
        ),
        user=env[
            "DB_USERNAME"
        ],
        password=env[
            "DB_PASSWORD"
        ],
    ) as conn:

        weight_column = (
            find_weight_column(
                conn
            )
        )

        print(
            "[WEIGHT COLUMN]",
            weight_column,
        )

        if (
            (
                min_weight is not None
                or max_weight is not None
            )
            and weight_column is None
        ):
            raise ValueError(
                "메뉴 중량 조건을 요청했지만 "
                "mealfit.menu 테이블에서 "
                "중량 컬럼을 찾을 수 없습니다."
            )

        if weight_column:
            weight_select = (
                f"m.{weight_column}"
            )
        else:
            weight_select = (
                "NULL::numeric"
            )

        sql = f"""
            SELECT
                m.menu_id,
                r.menu_code,
                m.name,
                m.upper_category,
                m.category,
                m.slot_type,
                {weight_select} AS weight_g,
                m.energy_kcal,
                m.protein_g,
                m.fat_g,
                m.carbohydrate_g,
                m.sodium_mg,
                r.embedding <=> %s::vector AS distance
            FROM mealfit.rag_document r
            JOIN mealfit.menu m
                ON r.menu_code = m.menu_code
            WHERE
                r.embedding_model = 'BAAI/bge-m3'
        """

        params = [
            vector
        ]

        # 구체 메뉴명 검색
        if menu_keyword:
            sql += """
                AND m.name ILIKE %s
            """

            params.append(
                f"%{menu_keyword}%"
            )

        # 메뉴 분류 검색
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

            params.append(
                slot
            )

        # 포함 식재료
        for ingredient in (
            include_ingredients
        ):
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

        # 제외 식재료
        for ingredient in (
            exclude_ingredients
        ):
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

        if (
            min_weight is not None
            and weight_column
        ):
            sql += f"""
                AND m.{weight_column} IS NOT NULL
                AND m.{weight_column} >= %s
            """

            params.append(
                float(
                    min_weight
                )
            )

        if (
            max_weight is not None
            and weight_column
        ):
            sql += f"""
                AND m.{weight_column} IS NOT NULL
                AND m.{weight_column} <= %s
            """

            params.append(
                float(
                    max_weight
                )
            )

        if min_calories is not None:
            sql += """
                AND m.energy_kcal IS NOT NULL
                AND m.energy_kcal >= %s
            """

            params.append(
                float(
                    min_calories
                )
            )

        if max_calories is not None:
            sql += """
                AND m.energy_kcal IS NOT NULL
                AND m.energy_kcal <= %s
            """

            params.append(
                float(
                    max_calories
                )
            )

        if min_protein is not None:
            sql += """
                AND m.protein_g IS NOT NULL
                AND m.protein_g >= %s
            """

            params.append(
                float(
                    min_protein
                )
            )

        if max_protein is not None:
            sql += """
                AND m.protein_g IS NOT NULL
                AND m.protein_g <= %s
            """

            params.append(
                float(
                    max_protein
                )
            )

        if min_sodium is not None:
            sql += """
                AND m.sodium_mg IS NOT NULL
                AND m.sodium_mg >= %s
            """

            params.append(
                float(
                    min_sodium
                )
            )

        if max_sodium is not None:
            sql += """
                AND m.sodium_mg IS NOT NULL
                AND m.sodium_mg <= %s
            """

            params.append(
                float(
                    max_sodium
                )
            )

        sql += """
            ORDER BY distance
            LIMIT 30
        """

        print(
            "[MENU KEYWORD]",
            menu_keyword,
        )

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
        weight_g,
        energy_kcal,
        protein_g,
        fat_g,
        carbohydrate_g,
        sodium_mg,
        distance,
    ) in rows:

        distance_float = float(
            distance
        )

        similarity = (
            1 - distance_float
        ) * 100

        print(
            f"[RAG] {name} | "
            f"distance={distance_float:.4f} | "
            f"similarity={similarity:.1f}%"
        )

        cost = menu_costs.get(
            menu_id
        )

        if similarity < 40:
            continue

        if (
            cost is None
            or cost <= 0
        ):
            continue

        results.append({
            "menu_id":
                menu_id,

            "menu_code":
                menu_code,

            "name":
                name,

            "main_category":
                upper_category,

            "sub_category":
                sub_category,

            "slot_type":
                slot_type,

            "cost_per_person": (
                float(cost)
                if cost is not None
                else None
            ),

            "weight_g": (
                float(weight_g)
                if weight_g is not None
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

            "distance":
                distance_float,

            "similarity":
                round(
                    similarity,
                    1,
                ),
        })

    # 유사도 높은 순 정렬
    results.sort(
        key=lambda menu:
            menu[
                "similarity"
            ],
        reverse=True,
    )

    return results[:30]