
import os
import json
import time
from langchain_community.document_loaders import PyPDFLoader
from langchain.vectorstores import Chroma, FAISS
from langchain_community.vectorstores.pgvector import PGVector
from langchain.text_splitter import CharacterTextSplitter
from ncp_embedding import  HCXEmbedding
from langchain.embeddings.openai import OpenAIEmbeddings



# __import__('pysqlite3')
# import sys
# sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')


ahn_asec_path = 'pdf data path'
db_save_path = "vector db save folder path" 
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

pdf_paths = []
for filename in os.listdir(ahn_asec_path):
    if filename.endswith(".pdf"):
        # 완전한 파일 경로 생성
        globals()[f'pdf_path_{filename}']  = os.path.join(ahn_asec_path, filename)
        # print(globals()[f'pdf_path_{filename}'])
        pdf_paths.append(globals()[f'pdf_path_{filename}'])

embeddings = HCXEmbedding()
# embeddings = OpenAIEmbeddings(model = 'text-embedding-3-small')
 
text_splitter = CharacterTextSplitter(        
                            separator = "\n",
                            chunk_size = 200,
                            chunk_overlap  = 50,
                            length_function = len,
                            )


def offline_chroma_save(pdf_paths):

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
            # print(len(single_embed_query))
            total_embed_query.append(single_embed_query)

    print(len(total_embed_query))
    print(len(total_docs))
    
    vectorstore = Chroma.from_documents(
        documents=total_docs,
        embedding=embeddings,
        persist_directory=os.path.join(ROOT_DIR, db_save_path, "cloud_bot_20240317_chroma_db")
        )
    vectorstore.persist()



CONNECTION_STRING = "postgresql+psycopg2://ID:PW@localhost:5432/DB NAME"
COLLECTION_NAME = "pgvector_db"
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
            # print(len(single_embed_query))
            total_embed_query.append(single_embed_query)

    print(len(total_embed_query))
    print(len(total_docs))
            
    vectorstore = PGVector.from_documents(
            documents=total_docs,            
            embedding=embeddings,
            collection_name=COLLECTION_NAME,
            connection_string=CONNECTION_STRING,
        )

 
start = time.time()
total_content = offline_chroma_save(pdf_paths)
total_content = offline_pgvector_save(pdf_paths)
end = time.time()
'''임베딩 완료 시간: 1.31 (초)'''
print('임베딩 완료 시간: %.2f (초)' %(end-start))
