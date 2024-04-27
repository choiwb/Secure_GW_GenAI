
import os
import uuid
from dotenv import load_dotenv

##################################################################################
# .env 파일 로드
load_dotenv()

API_KEY=os.getenv('HCX_API_KEY')
API_KEY_PRIMARY_VAL=os.getenv('HCX_API_KEY_PRIMARY_VAL')
REQUEST_ID=str(uuid.uuid4())

TOKEN_API_KEY=os.getenv('HCX_TOKEN_API_KEY')
##################################################################################

# pdf 파일 저장 경로
pdf_folder_path = os.path.join(os.getcwd(), 'YOUR PATH !!!!!!!')

# 임베딩 벡터 DB 저장 & 호출
db_save_path = os.path.join(os.getcwd(), 'YOUR PATH !!!!!!!') 

# asa, hcx 별 프로토콜 스택 이미지 경로
asa_image_path = os.path.join(os.getcwd(), 'YOUR PATH !!!!!!!')

# streamlit 아이콘 경로
you_icon = os.path.join(os.getcwd(), 'YOUR PATH !!!!!!!')
hcx_icon = os.path.join(os.getcwd(), 'YOUR PATH !!!!!!!')
ahn_icon = os.path.join(os.getcwd(), 'YOUR PATH !!!!!!!')
gpt_icon = os.path.join(os.getcwd(), 'YOUR PATH !!!!!!!')

# PostgreSQL 접속 정보
DB_CONNECTION_STRING = "DB_CONNECTION_STRING !!!!!!!"
DB_COLLECTION_NAME = "DB_COLLECTION_NAME !!!!!!!"

# sLLM 모델 경로
sllm_model_path = os.path.join(os.getcwd(), "sllm_models/EEVE-Korean-Instruct-10.8B-v1.0-Q4_K_M.gguf")
sllm_embed_model_path = os.path.join(os.getcwd(), "sllm_models/nomic-embed-text-v1.5.f32.gguf")

token_headers = {
        'X-NCP-CLOVASTUDIO-API-KEY': TOKEN_API_KEY,
        'X-NCP-APIGW-API-KEY': API_KEY_PRIMARY_VAL,
        'X-NCP-CLOVASTUDIO-REQUEST-ID': REQUEST_ID,
        'Content-Type': 'application/json; charset=utf-8',
        }

hcx_general_headers = {
        'X-NCP-CLOVASTUDIO-API-KEY': API_KEY,
        'X-NCP-APIGW-API-KEY': API_KEY_PRIMARY_VAL,
        'X-NCP-CLOVASTUDIO-REQUEST-ID': REQUEST_ID,
        'Content-Type': 'application/json; charset=utf-8',
        }

hcx_stream_headers = {
        'X-NCP-CLOVASTUDIO-API-KEY': API_KEY,
        'X-NCP-APIGW-API-KEY': API_KEY_PRIMARY_VAL,
        'X-NCP-CLOVASTUDIO-REQUEST-ID': REQUEST_ID,
        'Content-Type': 'application/json; charset=utf-8',
        # streaming 옵션 !!!!!
        'Accept': 'text/event-stream'
        }

# HCX & GPT & sLLM 주요 파라미터
llm_maxtokens = 512
llm_temperature = 0.1

sllm_n_gpu_layers = 1  # Metal set to 1 is enough.
sllm_n_batch = 8192  # Should be between 1 and n_ctx, consider the amount of RAM of your Apple Silicon Chip.
sllm_n_ctx = 8192
sllm_top_p = 1

hcx_llm_params = {
            'topP': 0.8,
            'topK': 0,
            'maxTokens': llm_maxtokens,
            'temperature': llm_temperature,
            'repeatPenalty': 5.0,
            'stopBefore': [],
            'includeAiFilters': True,
            "seed": 4595
            }

# 벡터 DB 관련 파라미터
db_search_type = 'mmr'
db_doc_k = 4
db_doc_fetch_k = 16
db_similarity_threshold = 0.5

