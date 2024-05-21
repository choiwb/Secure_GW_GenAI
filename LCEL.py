

import os
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
import streamlit as st
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from operator import itemgetter
from langchain_core.messages import get_buffer_string
from langchain.schema import format_document
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import EmbeddingsFilter
from langchain_community.vectorstores import Chroma

from config import db_save_path, db_name, user_db_name, DB_COLLECTION_NAME, DB_CONNECTION_STRING, db_search_type, db_doc_k, db_doc_fetch_k, db_similarity_threshold
from vector_db import embeddings
from prompt import not_rag_template, rag_template, img_rag_template, PROMPT_INJECTION_PROMPT, SYSTEMPROMPT
from LLM import HCX, gpt_model, sllm, gemini_vis_model, gemini_txt_model
from streamlit_custom_func import src_doc

ONLY_CHAIN_PROMPT = PromptTemplate(input_variables=["question"],template=not_rag_template)
QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context", "question"],template=rag_template)
IMG_QA_CHAIN_PROMPT = PromptTemplate(input_variables=["img_context", "context", "question"],template=img_rag_template)
 
hcx_sec = HCX(init_system_prompt = PROMPT_INJECTION_PROMPT, streaming = False)
hcx_stream = HCX(init_system_prompt = SYSTEMPROMPT, streaming = True)

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

@st.cache_resource
def asa_init_memory():
    return ConversationBufferMemory(
        input_key='question',
        output_key='answer',
        memory_key='chat_history',
        return_messages=True)
asa_memory = asa_init_memory()
 
@st.cache_resource
def hcx_init_memory():
    return ConversationBufferMemory(
        input_key='question',
        output_key='answer',
        memory_key='chat_history',
        return_messages=True)
hcx_memory = hcx_init_memory()
 
@st.cache_resource
def gpt_init_memory():
    return ConversationBufferMemory(
        input_key='question',
        output_key='answer',
        memory_key='chat_history',
        return_messages=True)
gpt_memory = gpt_init_memory()
 
@st.cache_resource
def sllm_init_memory():
    return ConversationBufferMemory(
        input_key='question',
        output_key='answer',
        memory_key='chat_history',
        return_messages=True)
sllm_memory = sllm_init_memory()

@st.cache_resource
def gemini_init_memory():
    return ConversationBufferMemory(
        input_key='question',
        output_key='answer',
        memory_key='chat_history',
        return_messages=True)
gemini_memory = gemini_init_memory()
 
# reset button
def reset_conversation():
    st.session_state.conversation = None
    st.session_state.chat_history = None
    asa_memory.clear()
    hcx_memory.clear()
    gpt_memory.clear()
    sllm_memory.clear()
    gemini_memory.clear()
    st.toast("대화 리셋!", icon="👋")

DEFAULT_DOCUMENT_PROMPT = PromptTemplate.from_template(template="{page_content}")
  
def _combine_documents(
    docs, document_prompt=DEFAULT_DOCUMENT_PROMPT, document_separator="\n\n"
):
    doc_strings = [format_document(doc, document_prompt) for doc in docs]
    return document_separator.join(doc_strings)
  
asa_loaded_memory = RunnablePassthrough.assign(
    chat_history=RunnableLambda(asa_memory.load_memory_variables) | itemgetter("chat_history"),
)

hcx_loaded_memory = RunnablePassthrough.assign(
    chat_history=RunnableLambda(hcx_memory.load_memory_variables) | itemgetter("chat_history"),
)
 
gpt_loaded_memory = RunnablePassthrough.assign(
    chat_history=RunnableLambda(gpt_memory.load_memory_variables) | itemgetter("chat_history"),
)

sllm_loaded_memory = RunnablePassthrough.assign(
    chat_history=RunnableLambda(sllm_memory.load_memory_variables) | itemgetter("chat_history"),
)

gemini_loaded_memory = RunnablePassthrough.assign(
    chat_history=RunnableLambda(gemini_memory.load_memory_variables) | itemgetter("chat_history"),
)
   
retrieved_documents = {
    "question": lambda x: x["question"],
    "source_documents": itemgetter("question") | compression_retriever,
    "chat_history": lambda x: get_buffer_string(x["chat_history"])
    }

user_retrieved_documents = {
    "question": lambda x: x["question"],
    "source_documents": itemgetter("question") | user_compression_retriever,
    "chat_history": lambda x: get_buffer_string(x["chat_history"])
    }

not_retrieved_documents = {
    "question": lambda x: x["question"],
    "chat_history": lambda x: get_buffer_string(x["chat_history"])
}

img_retrieved_documents = {
    "question": lambda x: x["question"],
    "img_context": lambda x: x["img_context"],
    "source_documents": itemgetter("question") | compression_retriever,
    "chat_history": lambda x: get_buffer_string(x["chat_history"])
}
 
final_inputs = {
    "context": lambda x: _combine_documents(x["source_documents"]),
    "question": itemgetter("question"),
    "chat_history": itemgetter("chat_history")
    }
 
img_final_inputs = {
    "img_context": itemgetter("img_context"),
    "context": lambda x: _combine_documents(x["source_documents"]),
    "question": itemgetter("question"),
    "chat_history": itemgetter("chat_history")
}

def RAG_pipeline(memory, documents, llm):
    return memory | documents | final_inputs | QA_CHAIN_PROMPT | src_doc | llm | StrOutputParser() 

def not_RAG_pipeline(memory, llm):
    return memory | not_retrieved_documents | ONLY_CHAIN_PROMPT | llm | StrOutputParser()

hcx_sec_pipe = ONLY_CHAIN_PROMPT | hcx_sec | StrOutputParser()
retrieval_qa_chain = RAG_pipeline(asa_loaded_memory, retrieved_documents, hcx_stream)
user_retrieval_qa_chain = RAG_pipeline(asa_loaded_memory, user_retrieved_documents, hcx_stream)
hcx_only_pipe =  not_RAG_pipeline(hcx_loaded_memory, hcx_stream)
gpt_pipe =  not_RAG_pipeline(gpt_loaded_memory, gpt_model)
sllm_pipe = RAG_pipeline(sllm_loaded_memory, retrieved_documents, sllm)

gemini_txt_pipe = not_RAG_pipeline(gemini_loaded_memory, gemini_txt_model)
gemini_vis_pipe = RunnablePassthrough() | gemini_vis_model | StrOutputParser()
gemini_vis_vectordb_txt_pipe = (
                                gemini_loaded_memory | img_retrieved_documents | img_final_inputs | 
                                IMG_QA_CHAIN_PROMPT | src_doc | gemini_txt_model | StrOutputParser()
                               )
