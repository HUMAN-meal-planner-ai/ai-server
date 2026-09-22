from pathlib import Path
import numpy as np
from FlagEmbedding import BGEM3FlagModel
from app.rag.menu_loader import load_menu_documents

#csv에서 전체 메뉴를 RAG 문서로 불러오기
documents = load_menu_documents("dataset/MENU_MASTER_v1.csv")

#메뉴 코드와 임베딩할 문장을 각각 준비
menu_codes = [doc["id"] for doc in documents]
texts = [doc["text"] for doc in documents]

#결과를 저장할 폴더와 파일 경로 
output_path = Path("outputs/rag/menu_embeddings.npz")
output_path.parent.mkdir(parents=True, exist_ok=True)

# 기존 결과를 실수로 덮어쓰지 않기
if output_path.exists():
    raise FileExistsError(f"이미 파일이 있습니다: {output_path}")

#BGE-M3 모델 불러오기
model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=False)

#전체 메뉴 문장을 벡터로 변환하기
result = model.encode(
    texts,
    batch_size=2,
    max_length=128,
    return_dense=True,
    return_sparse=False,
    return_colbert_vecs=False,
)

#메뉴 코드와 벡터를 함께 로컬 파일로 저장하기
vectors = np.asarray(result["dense_vecs"], dtype=np.float32)

np.savez_compressed(
    output_path,
    menu_codes=np.array(menu_codes),
    vectors=vectors,
)

print("전체 메뉴 수:", len(documents))
print("벡터 배열 크기:", vectors.shape)
print("저장 위치:", output_path)