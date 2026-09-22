import hashlib
import numpy as np
import psycopg
from dotenv import dotenv_values
from psycopg.types.json import Jsonb

from app.rag.menu_loader import load_menu_documents

# 백엔드의 DB 접속 정보 읽기
env = dotenv_values("../backend_new/.env")

# 기존 메뉴 문서와 임베딩 불러오기
docs = load_menu_documents("dataset/MENU_MASTER_v1.csv")

with np.load("outputs/rag/menu_embeddings.npz", allow_pickle=False) as data:
    codes = data["menu_codes"]
    vectors = data["vectors"]

# 메뉴 코드 순서와 벡터 크기 확인
if codes.tolist() != [doc["id"] for doc in docs] or vectors.shape != (len(docs), 1024):
    raise ValueError("메뉴 코드 또는 벡터 크기가 맞지 않습니다.")

print("저장할 메뉴:", len(docs))

if input("DB에 저장하려면 YES 입력: ") != "YES":
    raise SystemExit("저장 취소")

# 같은 메뉴 코드가 있으면 건너뛰고, 새 메뉴만 저장
sql = """
INSERT INTO mealfit.rag_document
(menu_code, content, metadata, embedding, embedding_model, content_hash, embedded_at)
VALUES (%s, %s, %s, %s::vector, %s, %s, NOW())
ON CONFLICT (menu_code) DO NOTHING
"""

rows = [
    (
        doc["id"],
        doc["text"],
        Jsonb(doc["metadata"]),
        "[" + ",".join(map(str, vector)) + "]",
        "BAAI/bge-m3",
        hashlib.sha256(doc["text"].encode()).hexdigest(),
    )
    for doc, vector in zip(docs, vectors)
]

with psycopg.connect(
    env["DB_URL"].removeprefix("jdbc:"),
    user=env["DB_USERNAME"],
    password=env["DB_PASSWORD"],
) as conn:
    with conn.cursor() as cur:
        cur.executemany(sql, rows)
        print("새로 저장한 문서:", cur.rowcount)

print("저장 완료!")