import json
from decimal import Decimal
from functools import lru_cache
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import psycopg
from dotenv import dotenv_values
from FlagEmbedding import BGEM3FlagModel

from app.rag.hybrid_search import bm25_search, rrf_fusion
from app.rag.query_parser import parse_query

ENV_PATH = Path(__file__).resolve().parents[2] / ".env"
TOP_K = 30


def get_env():
    return dotenv_values(ENV_PATH)


@lru_cache(maxsize=1)
def get_model():
    return BGEM3FlagModel("BAAI/bge-m3", use_fp16=False)


def get_menu_costs(authorization: str | None = None):
    env = get_env()
    backend_url = (env.get("BACKEND_URL") or "http://localhost:8080").rstrip("/")
    url = f"{backend_url}/api/cost/menus?{urlencode({'mealCount': 1})}"
    headers = {"Accept": "application/json"}
    if authorization:
        headers["Authorization"] = authorization

    request = Request(url, method="GET", headers=headers)
    try:
        with urlopen(request, timeout=15) as response:
            data = json.loads(response.read().decode("utf-8"))
    except HTTPError as e:
        if e.code == 401:
            raise RuntimeError(
                "원가 API 인증에 실패했습니다. 로그인 토큰이 전달되지 않았거나 유효하지 않습니다."
            ) from e
        raise RuntimeError(f"원가 API 호출 실패: HTTP {e.code}") from e
    except URLError as e:
        raise RuntimeError(f"원가 API에 연결할 수 없습니다: {url}") from e

    if isinstance(data, dict):
        data = data.get("data") or data.get("content") or data.get("items") or []
    if not isinstance(data, list):
        raise RuntimeError("원가 API 응답 형식을 확인할 수 없습니다.")

    costs = {}
    for item in data:
        menu_id = item.get("menuId")
        cost_per_person = item.get("costPerPerson")
        if menu_id is None or cost_per_person is None:
            continue
        costs[int(menu_id)] = Decimal(str(cost_per_person))
    return costs


def find_weight_column(conn):
    candidates = ["weight", "weight_g", "serving_weight_g", "standard_weight_g"]
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT column_name
            FROM information_schema.columns
            WHERE table_schema = 'mealfit' AND table_name = 'menu'
            """
        )
        existing = {row[0] for row in cur.fetchall()}
    return next((name for name in candidates if name in existing), None)


def search_menus(
    query: str,
    authorization: str | None = None,
    conditions: dict | None = None,
):
    if conditions is None:
        conditions = parse_query(query)

    print("[QUERY]", query)
    print("[CONDITIONS]", conditions)

    min_price = conditions.get("min_price")
    max_price = conditions.get("max_price")
    price_scope = conditions.get("price_scope")
    category = conditions.get("category")
    menu_keyword = conditions.get("menu_keyword")
    slot = conditions.get("slot")
    include_ingredients = conditions.get("include_ingredients") or []
    exclude_ingredients = conditions.get("exclude_ingredients") or []
    min_weight = conditions.get("min_weight")
    max_weight = conditions.get("max_weight")
    min_calories = conditions.get("min_calories")
    max_calories = conditions.get("max_calories")
    min_protein = conditions.get("min_protein")
    max_protein = conditions.get("max_protein")
    min_sodium = conditions.get("min_sodium")
    max_sodium = conditions.get("max_sodium")

    if (min_price is not None or max_price is not None) and price_scope == "meal":
        raise ValueError("식단 전체 예산 조건은 현재 메뉴 단위 검색에서는 처리할 수 없습니다.")

    menu_costs = get_menu_costs(authorization=authorization)
    # 원가가 없거나 0 이하인 메뉴는 두 검색 방식 모두에서 제외한다.
    allowed_menu_ids = []
    for menu_id, cost in menu_costs.items():
        if cost <= 0:
            continue
        if min_price is not None and cost < Decimal(str(min_price)):
            continue
        if max_price is not None and cost > Decimal(str(max_price)):
            continue
        allowed_menu_ids.append(menu_id)
    if not allowed_menu_ids:
        return []

    embedding = get_model().encode(
        [query],
        max_length=128,
        return_dense=True,
        return_sparse=False,
        return_colbert_vecs=False,
    )
    vector = "[" + ",".join(map(str, embedding["dense_vecs"][0])) + "]"
    env = get_env()

    with psycopg.connect(
        env["DB_URL"].removeprefix("jdbc:"),
        user=env["DB_USERNAME"],
        password=env["DB_PASSWORD"],
    ) as conn:
        weight_column = find_weight_column(conn)
        print("[WEIGHT COLUMN]", weight_column)
        if (min_weight is not None or max_weight is not None) and weight_column is None:
            raise ValueError(
                "메뉴 중량 조건을 요청했지만 mealfit.menu 테이블에서 중량 컬럼을 찾을 수 없습니다."
            )

        # 컬럼명은 DB 메타데이터의 허용 목록에서만 선택된다.
        weight_select = f"m.{weight_column}" if weight_column else "NULL::numeric"
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
                CONCAT_WS(
                    ' ',
                    m.name,
                    m.upper_category,
                    m.category,
                    m.slot_type,
                    (
                        SELECT STRING_AGG(i.name, ' ')
                        FROM mealfit.menu_ingredient mi
                        JOIN mealfit.ingredient i
                            ON i.ingredient_id = mi.ingredient_id
                        WHERE mi.menu_id = m.menu_id
                    )
                ) AS search_text,
                r.embedding <=> %s::vector AS distance
            FROM mealfit.rag_document r
            JOIN mealfit.menu m ON r.menu_code = m.menu_code
            WHERE r.embedding_model = 'BAAI/bge-m3'
              AND m.menu_id = ANY(%s::bigint[])
        """
        params = [vector, allowed_menu_ids]

        if menu_keyword:
            sql += " AND m.name ILIKE %s"
            params.append(f"%{menu_keyword}%")
        if category:
            sql += " AND (m.upper_category ILIKE %s OR m.category ILIKE %s)"
            params.extend([f"%{category}%", f"%{category}%"])
        if slot:
            sql += " AND m.slot_type = %s"
            params.append(slot)

        for ingredient in include_ingredients:
            sql += """
                AND EXISTS (
                    SELECT 1
                    FROM mealfit.menu_ingredient mi_inc
                    JOIN mealfit.ingredient i_inc
                        ON i_inc.ingredient_id = mi_inc.ingredient_id
                    WHERE mi_inc.menu_id = m.menu_id
                      AND i_inc.name ILIKE %s
                )
            """
            params.append(f"%{ingredient}%")
        for ingredient in exclude_ingredients:
            sql += """
                AND NOT EXISTS (
                    SELECT 1
                    FROM mealfit.menu_ingredient mi_exc
                    JOIN mealfit.ingredient i_exc
                        ON i_exc.ingredient_id = mi_exc.ingredient_id
                    WHERE mi_exc.menu_id = m.menu_id
                      AND i_exc.name ILIKE %s
                )
            """
            params.append(f"%{ingredient}%")

        numeric_conditions = [
            (min_weight, weight_column, ">="),
            (max_weight, weight_column, "<="),
            (min_calories, "energy_kcal", ">="),
            (max_calories, "energy_kcal", "<="),
            (min_protein, "protein_g", ">="),
            (max_protein, "protein_g", "<="),
            (min_sodium, "sodium_mg", ">="),
            (max_sodium, "sodium_mg", "<="),
        ]
        for value, column, op in numeric_conditions:
            if value is not None and column:
                sql += f" AND m.{column} IS NOT NULL AND m.{column} {op} %s"
                params.append(float(value))

        # 벡터 순위와 무관하게 모든 조건 통과 후보를 가져온다.
        # 벡터와 BM25가 동일한 후보 집합을 독립적으로 순위화한다.
        sql += " ORDER BY distance ASC"
        with conn.cursor() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()

    results = []
    for (
        menu_id, menu_code, name, upper_category, sub_category, slot_type,
        weight_g, energy_kcal, protein_g, fat_g, carbohydrate_g,
        sodium_mg, search_text, distance,
    ) in rows:
        if distance is None:
            continue
        distance_float = float(distance)
        similarity = (1 - distance_float) * 100
        cost = menu_costs.get(menu_id)
        if cost is None or cost <= 0:
            continue

        results.append({
            "menu_id": menu_id,
            "menu_code": menu_code,
            "name": name,
            "main_category": upper_category,
            "sub_category": sub_category,
            "slot_type": slot_type,
            "cost_per_person": float(cost),
            "weight_g": float(weight_g) if weight_g is not None else None,
            "energy_kcal": float(energy_kcal) if energy_kcal is not None else None,
            "protein_g": float(protein_g) if protein_g is not None else None,
            "fat_g": float(fat_g) if fat_g is not None else None,
            "carbohydrate_g": float(carbohydrate_g) if carbohydrate_g is not None else None,
            "sodium_mg": float(sodium_mg) if sodium_mg is not None else None,
            "distance": distance_float,
            "similarity": round(similarity, 1),
            "search_text": search_text or "",
        })

    if not results:
        return []

    vector_docs = [
        {"id": menu["menu_id"], "text": menu["search_text"]}
        for menu in results if menu["similarity"] >= 40
    ][:TOP_K]
    documents = [
        {"id": menu["menu_id"], "text": menu["search_text"]}
        for menu in results
    ]
    bm25_docs = bm25_search(query, documents, top_k=TOP_K)
    fused = rrf_fusion(vector_docs, bm25_docs)
    by_id = {str(menu["menu_id"]): menu for menu in results}

    print(f"[HYBRID] candidates={len(results)} vector={len(vector_docs)} bm25={len(bm25_docs)}")

    # search_text는 내부 BM25 용도로만 사용하며 기존 API 응답 형식을 유지한다.
    final_results = []
    for document in fused[:TOP_K]:
        item = by_id[str(document["id"])].copy()
        item.pop("search_text", None)
        final_results.append(item)
    return final_results
