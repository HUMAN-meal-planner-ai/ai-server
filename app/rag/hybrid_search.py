
import re
from rank_bm25 import BM25Okapi


def tokenize(text: str) -> list[str]:
    """한국어 부분 일치를 위한 간단한 토큰화"""
    words = re.findall(r"[가-힣]+|[a-zA-Z0-9]+", text.lower())
    tokens = []

    for word in words:
        tokens.append(word)

        if re.search(r"[가-힣]", word) and len(word) >= 2:
            tokens.extend(
                word[i:i + 2]
                for i in range(len(word) - 1)
            )

    return tokens


def bm25_search(query: str, documents: list[dict], top_k=30):
    """메뉴 문서를 BM25 점수순으로 검색"""
    if not documents or not query.strip():
        return []

    corpus = [tokenize(doc["text"]) for doc in documents]
    model = BM25Okapi(corpus)
    scores = model.get_scores(tokenize(query))

    ranked = sorted(
        range(len(documents)),
        key=lambda i: scores[i],
        reverse=True,
    )

    return [
        documents[i]
        for i in ranked[:top_k]
        if scores[i] > 0
    ]


def rrf_fusion(vector_results, bm25_results, k=60):
    """Vector와 BM25 검색 순위를 RRF로 통합"""
    scores = {}
    documents = {}

    for results in (vector_results, bm25_results):
        for rank, doc in enumerate(results, start=1):
            menu_id = str(doc["id"])
            documents[menu_id] = doc
            scores[menu_id] = (
                scores.get(menu_id, 0)
                + 1 / (k + rank)
            )

    ranked_ids = sorted(
        scores,
        key=lambda menu_id: scores[menu_id],
        reverse=True,
    )

    return [documents[menu_id] for menu_id in ranked_ids]
