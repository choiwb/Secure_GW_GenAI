
import os
import numpy as np
from dotenv import load_dotenv
from langchain_community.embeddings import ClovaEmbeddings

###################################################################
load_dotenv()

CLOVA_EMB_API_KEY = os.getenv("CLOVA_EMB_API_KEY")
CLOVA_EMB_APIGW_API_KEY = os.getenv("CLOVA_EMB_APIGW_API_KEY")
EMBEDDING_APP_ID = os.getenv("EMBEDDING_APP_ID")
###################################################################

def cosine_similarity(A, B):
    """## 코사인 유사도 계산

    ### Args:
        - `A (vector)`: text1의 임베딩 벡터
        - `B (vector)`: text2의 임베딩 벡터

    ### Returns:
        - `float`: 유사도 점수
    """
    return np.dot(A, B)/(np.linalg.norm(A)*np.linalg.norm(B))

def euclidean_distance(A, B):
    """## 유클리디안 거리 계산

    ### Args:
        - `A (vector)`: text1의 임베딩 벡터
        - `B (vector)`: text2의 임베딩 벡터

    ### Returns:
        - `float`: 거리 점수
    """
    A = np.array(A)
    B = np.array(B)
    return np.linalg.norm(A - B)

embedding_emb = ClovaEmbeddings(
    model = 'clir-emb-dolphin',
    clova_emb_api_key=CLOVA_EMB_API_KEY,
    clova_emb_apigw_api_key=CLOVA_EMB_APIGW_API_KEY,
    app_id=EMBEDDING_APP_ID
)

embedding_sts = ClovaEmbeddings(
    model = 'clir-sts-dolphin',
    clova_emb_api_key=CLOVA_EMB_API_KEY,
    clova_emb_apigw_api_key=CLOVA_EMB_APIGW_API_KEY,
    app_id=EMBEDDING_APP_ID
)

query_text_1 = "오늘 날씨 어때?"
query_text_2 = "오늘은 비가 옵니다."
# query_text_2 = "날씨 오늘 어떨까?"

sts_query_result_1 = embedding_sts.embed_query(query_text_1)
sts_query_result_2 = embedding_sts.embed_query(query_text_2)

cosine_score_sts = cosine_similarity(sts_query_result_1, sts_query_result_2)
print('clir-sts-dolphin 모델의 코사인 유사도: %.2f' %(cosine_score_sts))

emb_query_result_1 = embedding_emb.embed_query(query_text_1)
emb_query_result_2 = embedding_emb.embed_query(query_text_2)
euclidean_score_emb = euclidean_distance(emb_query_result_1, emb_query_result_2)
print('clir-emb-dolphin 모델의 유클리드 거리 (L2 거리): %.2f' %(euclidean_score_emb))

