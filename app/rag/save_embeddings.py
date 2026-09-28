import hashlib
from pathlib import Path

import numpy as np
import psycopg
from dotenv import dotenv_values
from psycopg.types.json import Jsonb

from app.rag.menu_loader import load_menu_documents_from_db


# 경로 설정
AI_SERVER_ROOT = Path(__file__).resolve().parents[2]

ENV_PATH = (
    Path(__file__).resolve().parents[3]
    / "backend_new"
    / ".env"
)

EMBEDDING_PATH = (
    AI_SERVER_ROOT
    / "outputs"
    / "rag"
    / "menu_embeddings.npz"
)


# DB 접속 정보 읽기
env = dotenv_values(ENV_PATH)

db_url = env.get("DB_URL")
db_username = env.get("DB_USERNAME")
db_password = env.get("DB_PASSWORD")

if not db_url or not db_username or not db_password:
    raise ValueError(
        "DB 접속 정보를 .env에서 찾을 수 없습니다."
    )


# DB에서 메뉴 + 식재료 문서 불러오기
print("메뉴 문서를 DB에서 불러오는 중...")

docs = load_menu_documents_from_db()

print("불러온 메뉴 수:", len(docs))


# BGE-M3 임베딩 파일 불러오기
if not EMBEDDING_PATH.exists():
    raise FileNotFoundError(
        f"임베딩 파일이 없습니다: {EMBEDDING_PATH}"
    )

with np.load(
    EMBEDDING_PATH,
    allow_pickle=False,
) as data:
    codes = data["menu_codes"]
    vectors = data["vectors"]


# 메뉴 코드와 임베딩 검증
document_codes = [
    doc["id"]
    for doc in docs
]

if codes.tolist() != document_codes:
    raise ValueError(
        "RAG 문서와 임베딩 파일의 메뉴 코드 순서가 다릅니다."
    )

if vectors.shape != (len(docs), 1024):
    raise ValueError(
        f"벡터 크기가 맞지 않습니다: {vectors.shape}"
    )

print("저장할 메뉴:", len(docs))
print("벡터 크기:", vectors.shape)


# DB 저장 여부 확인
answer = input(
    "DB의 RAG 데이터를 갱신하려면 YES 입력: "
)

if answer != "YES":
    raise SystemExit("저장을 취소했습니다.")


# RAG 문서 저장 SQL
sql = """
INSERT INTO mealfit.rag_document
(
    menu_code,
    content,
    metadata,
    embedding,
    embedding_model,
    content_hash,
    embedded_at
)
VALUES (
    %s,
    %s,
    %s,
    %s::vector,
    %s,
    %s,
    NOW()
)

ON CONFLICT (menu_code)
DO UPDATE SET
    content = EXCLUDED.content,
    metadata = EXCLUDED.metadata,
    embedding = EXCLUDED.embedding,
    embedding_model = EXCLUDED.embedding_model,
    content_hash = EXCLUDED.content_hash,
    embedded_at = NOW()
"""


# DB에 저장할 데이터 만들기
rows = []

for doc, vector in zip(docs, vectors):
    vector_text = (
        "["
        + ",".join(map(str, vector))
        + "]"
    )

    content_hash = hashlib.sha256(
        doc["text"].encode("utf-8")
    ).hexdigest()

    rows.append(
        (
            doc["id"],
            doc["text"],
            Jsonb(doc["metadata"]),
            vector_text,
            "BAAI/bge-m3",
            content_hash,
        )
    )


# DB에 RAG 문서 저장
print("RAG 문서를 DB에 저장하는 중...")

with psycopg.connect(
    db_url.removeprefix("jdbc:"),
    user=db_username,
    password=db_password,
) as conn:
    with conn.cursor() as cur:
        cur.executemany(
            sql,
            rows,
        )


print("RAG DB 갱신 완료")
print("메뉴 수:", len(docs))
print("임베딩 모델: BAAI/bge-m3")