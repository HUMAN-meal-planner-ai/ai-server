import psycopg
from pathlib import Path
from dotenv import dotenv_values
from FlagEmbedding import BGEM3FlagModel

# 백엔드의 기존 DB 접속 정보 읽기
env_path = Path(__file__).resolve().parents[3] / "backend_new" / ".env"
env = dotenv_values(env_path)

# 메뉴 임베딩에 사용한 모델과 동일한 모델 사용
model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=False)

# 사용자가 검색어 입력
query = input("검색어를 입력하세요: ")

# 검색어를 1024차원 벡터로 변환
result = model.encode(
    [query],
    max_length=128,
    return_dense=True,
    return_sparse=False,
    return_colbert_vecs=False,
)

vector = result["dense_vecs"][0]
vector_text = "[" + ",".join(map(str, vector)) + "]"

# DB에 저장된 메뉴 벡터와 유사도 비교
sql = """
SELECT menu_code, content,
       embedding <=> %s::vector AS distance
FROM mealfit.rag_document
WHERE embedding_model = 'BAAI/bge-m3'
ORDER BY distance
LIMIT 5
"""

with psycopg.connect(
    env["DB_URL"].removeprefix("jdbc:"),
    user=env["DB_USERNAME"],
    password=env["DB_PASSWORD"],
) as conn:
    with conn.cursor() as cur:
        cur.execute(sql, (vector_text,))
        menus = cur.fetchall()

# 검색 결과 출력
print("\n검색 결과 TOP 5")
for menu_code, content, distance in menus:
    print(f"\n[{menu_code}] {content}")
    print(f"벡터 거리: {distance:.4f}")