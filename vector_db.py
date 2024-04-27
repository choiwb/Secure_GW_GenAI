
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_community.embeddings import LlamaCppEmbeddings
from langchain_community.vectorstores.pgvector import PGVector

from ncp_embedding import HCXEmbedding
from config import db_save_path, sllm_embed_model_path, DB_COLLECTION_NAME, DB_CONNECTION_STRING


# text-embedding-3-small or text-embedding-3-large
# embeddings = OpenAIEmbeddings(model = 'text-embedding-3-small')
# embeddings = LlamaCppEmbeddings(model_path = sllm_embed_model_path)
embeddings = HCXEmbedding()

 
text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=200, chunk_overlap=50, is_separator_regex=False
    )

# 오프라인 데이터 가공 ####################################################################################
# OpenAI 또는 HuggingFace Embedding
def offline_chroma_save(pdf_paths):
 
    total_docs = []
   
    for pdf_url in pdf_paths:
        pdfreader =  PyPDFLoader(pdf_url)
        pdf_doc = pdfreader.load_and_split()
        doc = text_splitter.split_documents(pdf_doc)
        total_docs = total_docs + doc

    vectorstore = Chroma.from_documents(
        documents=total_docs,
        embedding=embeddings,
        persist_directory=os.path.join(db_save_path, "cloud_assistant_v4")
        )
    vectorstore.persist()


# NCP Embedding
def ncp_offline_chroma_save(pdf_paths):

    total_docs = []
    total_embed_query = []
    
    for pdf_url in pdf_paths:
        pdfreader =  PyPDFLoader(pdf_url)
        pdf_doc = pdfreader.load_and_split()
        doc = text_splitter.split_documents(pdf_doc)
        total_docs = total_docs + doc

        page_content = [doc.page_content for doc in doc]

        for content in page_content:
            single_embed_query = embeddings.embed_query(content)
            # 1024 차원
            # print(len(single_embed_query))
            total_embed_query.append(single_embed_query)
    
    vectorstore = Chroma.from_documents(
        documents=total_docs,
        embedding=embeddings,
        persist_directory=os.path.join(db_save_path, "cloud_assistant_v3")
        )
    vectorstore.persist()

'''
postgresql 설치 후, pgvector 설치해야 함.
'''
def offline_pgvector_save(pdf_paths):

    total_docs = []
    total_embed_query = []
    
    for pdf_url in pdf_paths:
        pdfreader =  PyPDFLoader(pdf_url)
        pdf_doc = pdfreader.load_and_split()
        doc = text_splitter.split_documents(pdf_doc)
        total_docs = total_docs + doc
        
        page_content = [doc.page_content for doc in doc]
        
        for content in page_content:
            single_embed_query = embeddings.embed_query(content)
            # 1024 차원
            # print(len(single_embed_query))
            total_embed_query.append(single_embed_query)

    print(len(total_embed_query))
    print(len(total_docs))
            
    vectorstore = PGVector.from_documents(
            documents=total_docs,            
            embedding=embeddings,
            collection_name=DB_COLLECTION_NAME,
            connection_string=DB_CONNECTION_STRING,
        )
 #######################################################################################################
