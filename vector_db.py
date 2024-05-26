
import os
import boto3
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import LlamaCppEmbeddings
from langchain_community.vectorstores.pgvector import PGVector
from langchain_aws import BedrockEmbeddings

from ncp_embedding import HCXEmbedding
from config import aws_embed_model_id, db_save_path, sllm_embed_model_path, DB_COLLECTION_NAME, DB_CONNECTION_STRING


##################################################################################
# .env 파일 로드
load_dotenv()

os.getenv('OPENAI_API_KEY')

# aws 콘솔 - IAM 사용자 그룹 및 사용자 생성
# Bedrock 접근 정책 생성
# 해당 사용자에 위 접근 정책 부여
# aws configure
# vi ~/.aws/credentials

AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_REGION")
##################################################################################
    
# Bedrock Runtime
bedrock_runtime = boto3.client(
    service_name="bedrock-runtime",
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,     
    region_name=AWS_REGION
)

# text-embedding-3-small or text-embedding-3-large
# embeddings = OpenAIEmbeddings(model = 'text-embedding-3-small')
# embeddings = LlamaCppEmbeddings(model_path = sllm_embed_model_path)
# embeddings = HCXEmbedding()
embeddings = BedrockEmbeddings(
        client=bedrock_runtime,
        region_name=AWS_REGION,
        model_id=aws_embed_model_id
    ) 
 
text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=200, chunk_overlap=50, is_separator_regex=False
    )

# 오프라인 데이터 가공 ####################################################################################
# OpenAI, NCP, Amazon Titan, HuggingFace Embedding
def offline_chroma_save(pdf_paths, db_name): 
    total_docs = []
    for pdf_url in pdf_paths:
        pdfreader =  PyPDFLoader(pdf_url)
        pdf_doc = pdfreader.load_and_split()

        doc = text_splitter.split_documents(pdf_doc)
        total_docs = total_docs + doc
    
    vectorstore = Chroma.from_documents(
        documents=total_docs,
        embedding=embeddings,
        persist_directory=os.path.join(db_save_path, db_name)
        )
    vectorstore.persist()
    
# postgresql 설치 후, pgvector 설치해야 함.
def offline_pgvector_save(pdf_paths):
    total_docs = []
    for pdf_url in pdf_paths:
        pdfreader =  PyPDFLoader(pdf_url)
        pdf_doc = pdfreader.load_and_split()
        doc = text_splitter.split_documents(pdf_doc)
        total_docs = total_docs + doc
                            
    vectorstore = PGVector.from_documents(
            documents=total_docs,            
            embedding=embeddings,
            collection_name=DB_COLLECTION_NAME,
            connection_string=DB_CONNECTION_STRING,
        )
#######################################################################################################
