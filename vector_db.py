
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_milvus import Milvus
from langchain_community.vectorstores.pgvector import PGVector

from config import db_save_path, DB_COLLECTION_NAME, DB_CONNECTION_STRING, embeddings

 
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
    
def offline_chroma_update(pdf_paths, db_name):
    # 기존 데이터베이스 로드
    vectorstore = Chroma(
        embedding=embeddings,
        persist_directory=os.path.join(db_save_path, db_name)
    )
    
    # 새로운 문서 로드 및 벡터화
    total_docs = []
    for pdf_url in pdf_paths:
        pdfreader = PyPDFLoader(pdf_url)
        pdf_doc = pdfreader.load_and_split()

        doc = text_splitter.split_documents(pdf_doc)
        total_docs += doc
    
    # 벡터 DB 업데이트
    vectorstore.add_documents(documents=total_docs)
    
    # 변경 사항 저장
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
    
def offline_milvus_save(pdf_paths): 
    total_docs = []
    for pdf_url in pdf_paths:
        pdfreader =  PyPDFLoader(pdf_url)
        pdf_doc = pdfreader.load_and_split()

        doc = text_splitter.split_documents(pdf_doc)
        total_docs = total_docs + doc
    
    vectorstore = Milvus.from_documents(
            documents=total_docs,
            embedding=embeddings,
            connection_args={"uri": os.path.join(db_save_path, 'cloud_assistant_v5_milvus.db')},
            drop_old=True,  # Drop the old Milvus collection if it exists
            )
    
async def async_offline_milvus_save(pdf_paths): 
    total_docs = []
    for pdf_url in pdf_paths:
        pdfreader =  PyPDFLoader(pdf_url)
        pdf_doc = pdfreader.load_and_split()

        doc = text_splitter.split_documents(pdf_doc)
        total_docs = total_docs + doc

    print(len(total_docs))

    vectorstore = await Milvus.afrom_documents(
            documents=total_docs,
            embedding=embeddings,
            connection_args={"uri": os.path.join(db_save_path, 'nia_poc_sample_milvus.db')},
            drop_old=True,  # Drop the old Milvus collection if it exists
            )
    
    return vectorstore
#######################################################################################################
