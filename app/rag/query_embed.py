from FlagEmbedding import BGEM3FlagModel

# 메뉴 임베딩에 사용했던 동일한 모델
model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=False)

# 사용자가 검색할 문장
query = "따뜻한 국 메뉴"

# 검색어를 벡터로 변환
result = model.encode(
    [query],
    return_dense=True,
    return_sparse=False,
    return_colbert_vecs=False,
)

vector = result["dense_vecs"][0]

print("검색어:", query)
print("벡터 차원:", len(vector))