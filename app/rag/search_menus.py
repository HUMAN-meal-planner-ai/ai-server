from functools import lru_cache
from pathlib import Path

import psycopg
from dotenv import dotenv_values
from FlagEmbedding import BGEM3FlagModel

from app.rag.query_parser import parse_query


@lru_cache(maxsize=1)
def get_model():
    return BGEM3FlagModel("BAAI/bge-m3", use_fp16=False)


def search_menus(query: str):
    conditions = parse_query(query)
    category = conditions.get("category")

    if conditions.get("max_price") is not None:
        raise ValueError("가격 조건 검색은 메뉴별 원가 데이터 연결 후 가능합니다.")

    result = get_model().encode(
        [query],
        max_length=128,
        return_dense=True,
        return_sparse=False,
        return_colbert_vecs=False,
    )
    vector = "[" + ",".join(map(str, result["dense_vecs"][0])) + "]"

    sql = """
        SELECT r.menu_code, m.name, m.upper_category, m.category,
               r.embedding <=> %s::vector AS distance
        FROM mealfit.rag_document r
        JOIN mealfit.menu m ON r.menu_code = m.menu_code
        WHERE r.embedding_model = 'BAAI/bge-m3'
    """
    params = [vector]

    if category:
        sql += " AND (m.upper_category ILIKE %s OR m.category ILIKE %s)"
        params.extend([f"%{category}%", f"%{category}%"])

    sql += " ORDER BY distance LIMIT 8"

    env_path = Path(__file__).resolve().parents[3] / "backend_new" / ".env"
    env = dotenv_values(env_path)

    with psycopg.connect(
        env["DB_URL"].removeprefix("jdbc:"),
        user=env["DB_USERNAME"],
        password=env["DB_PASSWORD"],
    ) as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()

    return [
        {
            "menu_code": code,
            "name": name,
            "main_category": main,
            "sub_category": sub,
            "distance": float(distance),
        }
        for code, name, main, sub, distance in rows
    ]


if __name__ == "__main__":
    query = input("검색어를 입력하세요: ")
    for menu in search_menus(query):
        print(menu)