
import os
import boto3
import uuid
from dotenv import load_dotenv
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import EmbeddingsFilter
from langchain_community.vectorstores import Chroma
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from langchain_openai import OpenAIEmbeddings
from langchain_aws import BedrockEmbeddings
from langchain_community.embeddings import ClovaEmbeddings, LlamaCppEmbeddings
from langchain_community.vectorstores.pgvector import PGVector

##################################################################################
# .env 파일 로드
load_dotenv()

API_KEY=os.getenv('HCX_API_KEY')
API_KEY_PRIMARY_VAL=os.getenv('HCX_API_KEY_PRIMARY_VAL')
REQUEST_ID=str(uuid.uuid4())

TOKEN_API_KEY=os.getenv('HCX_TOKEN_API_KEY')

CLOVA_EMB_API_KEY = os.getenv("CLOVA_EMB_API_KEY")
CLOVA_EMB_APIGW_API_KEY = os.getenv("CLOVA_EMB_APIGW_API_KEY")
EMBEDDING_APP_ID = os.getenv("EMBEDDING_APP_ID")

os.getenv('OPENAI_API_KEY')

# HCX LLM 경로 !!!!!!!!!!!!!!!!!!!!!!!
HCX_LLM_URL = os.getenv('HCX_LLM_URL')

# Set Google API key
os.getenv("GOOGLE_API_KEY")
os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

# aws 콘솔 - IAM 사용자 그룹 및 사용자 생성
# Bedrock 접근 정책 생성
# 해당 사용자에 위 접근 정책 부여
# aws configure
# vi ~/.aws/credentials

AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_REGION")
##################################################################################

# pdf 파일 저장 경로
pdf_folder_path = os.path.join(os.getcwd(), 'YOUR PATH !!!!!!!')
user_pdf_folder_path = os.path.join(os.getcwd(), 'YOUR PATH !!!!!!!')
db_name = 'INIT VECTOR DB NAME !!!!!!!'
user_db_name = 'USER VECTOR DB NAME !!!!!!!'

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

# Bedrock Runtime
bedrock_runtime = boto3.client(
    service_name="bedrock-runtime",
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,     
    region_name=AWS_REGION
)

# AWS 임베딩 & LLM 모델
aws_embed_model_id = "amazon.titan-embed-text-v1"
aws_llm_id = "anthropic.claude-3-5-sonnet-20240620-v1:0"

# text-embedding-3-small or text-embedding-3-large
# embeddings = OpenAIEmbeddings(model = 'text-embedding-3-small')
# embeddings = LlamaCppEmbeddings(model_path = sllm_embed_model_path)
# embeddings = ClovaEmbeddings(
#         model = 'clir-sts-dolphin',
#         clova_emb_api_key=CLOVA_EMB_API_KEY,
#         clova_emb_apigw_api_key=CLOVA_EMB_APIGW_API_KEY,
#         app_id=EMBEDDING_APP_ID
#         )
embeddings = BedrockEmbeddings(
        client=bedrock_runtime,
        region_name=AWS_REGION,
        model_id=aws_embed_model_id
        ) 

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
db_similarity_threshold = 0.4

# new_docsearch = PGVector(collection_name=DB_COLLECTION_NAME, connection_string=DB_CONNECTION_STRING,
#                         embedding_function=embeddings)

new_docsearch = Chroma(persist_directory=os.path.join(db_save_path, db_name),
                            embedding_function=embeddings)
user_new_docsearch = Chroma(persist_directory=os.path.join(db_save_path, user_db_name),
                            embedding_function=embeddings)

def retriever_alog(new_docsearch):
    retriever = new_docsearch.as_retriever(
                                        search_type=db_search_type,         
                                        search_kwargs={'k': db_doc_k, 'fetch_k': db_doc_fetch_k}
                                    )
    embeddings_filter = EmbeddingsFilter(embeddings=embeddings, similarity_threshold=db_similarity_threshold) 
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=embeddings_filter, base_retriever=retriever
    )
    return compression_retriever

compression_retriever = retriever_alog(new_docsearch)
user_compression_retriever = retriever_alog(user_new_docsearch)

gemini_llm_params = {
        'temperature': llm_temperature,
        'max_output_tokens': llm_maxtokens
}

gemini_safe = {
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE
}


