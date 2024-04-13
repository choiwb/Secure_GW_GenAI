
import os

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
sllm_n_gpu_layers = 1  # Metal set to 1 is enough.
sllm_n_batch = 8192  # Should be between 1 and n_ctx, consider the amount of RAM of your Apple Silicon Chip.

