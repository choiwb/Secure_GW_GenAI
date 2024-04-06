
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings

from ncp_embedding import HCXEmbedding
from config import db_save_path


# text-embedding-3-small or text-embedding-3-large
embeddings = OpenAIEmbeddings(model = 'text-embedding-3-small')
# embeddings = HCXEmbedding()

 
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=100, chunk_overlap=20,
        is_separator_regex=False,
    )

# 오프라인 데이터 가공 ####################################################################################
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
        persist_directory=os.path.join(db_save_path, "cloud_assistant_v1")
        )
    vectorstore.persist()
 #######################################################################################################
