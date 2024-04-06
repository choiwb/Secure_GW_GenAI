

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
from langchain.vectorstores import Chroma

from config import db_save_path, DB_COLLECTION_NAME, DB_CONNECTION_STRING
from vector_db import embeddings
from prompt import not_rag_template, rag_template
from LLM import HCX_sec, HCX_only, HCX_stream, gpt_model, gemini_vis_model, gemini_txt_model


 
ONLY_CHAIN_PROMPT = PromptTemplate(input_variables=["question"],template=not_rag_template)
SEC_CHAIN_PROMPT = PromptTemplate(input_variables=["question"],template=not_rag_template)
QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context", "question"],template=rag_template)
 
hcx_sec = HCX_sec()
hcx_stream = HCX_stream()
hcx_only = HCX_only() 
 
new_docsearch = Chroma(persist_directory=os.path.join(db_save_path, "cloud_assistant_v1"),
                        embedding_function=embeddings)

# new_docsearch = PGVector(collection_name=DB_COLLECTION_NAME, connection_string=DB_CONNECTION_STRING,
#                         embedding_function=embeddings)

retriever = new_docsearch.as_retriever(
                                        search_type="mmr",                                        
                                        search_kwargs={'k': 4, 'fetch_k': 16}
                                       )


# retriever의 compression 시도 !!!!!!!!!!!!!!!!!!!!!!!!!
'''
ncp embedding의 경우
ValidationError: 1 validation error for EmbeddingsFilter embeddings instance of Embeddings expected (type=type_error.arbitrary_type; expected_arbitrary_type=Embeddings)
'''
embeddings_filter = EmbeddingsFilter(embeddings=embeddings, similarity_threshold=0.3) 

compression_retriever = ContextualCompressionRetriever(
    base_compressor=embeddings_filter, base_retriever=retriever
)
 
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
def gemini_init_memory():
    return ConversationBufferMemory(
        input_key='question',
        output_key='answer',
        memory_key='chat_history',
        return_messages=True)
gemini_memory = gemini_init_memory()
 
# reset button
def reset_conversation():
  print('대화 리셋!')
  st.session_state.conversation = None
  st.session_state.chat_history = None
  asa_memory.clear()
  hcx_memory.clear()
  gpt_memory.clear()
  gemini_memory.clear()
  
  

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

gemini_loaded_memory = RunnablePassthrough.assign(
    chat_history=RunnableLambda(gemini_memory.load_memory_variables) | itemgetter("chat_history"),
)
  
retrieved_documents = {
    "question": lambda x: x["question"],
    "source_documents": itemgetter("question") | compression_retriever,
    "chat_history": lambda x: get_buffer_string(x["chat_history"])
}
 
not_retrieved_documents = {
    "question": lambda x: x["question"],
    "chat_history": lambda x: get_buffer_string(x["chat_history"])
}

img_retrieved_documents = {
    "context": lambda x: x["context"],
    "question": lambda x: x["question"]
}
 
final_inputs = {
    "context": lambda x: _combine_documents(x["source_documents"]),
    "question": itemgetter("question"),
    "chat_history": itemgetter("chat_history")
}


hcx_sec_pipe = SEC_CHAIN_PROMPT | hcx_sec | StrOutputParser()
retrieval_qa_chain =  asa_loaded_memory | retrieved_documents | final_inputs | QA_CHAIN_PROMPT | hcx_stream | StrOutputParser()
hcx_only_pipe =  hcx_loaded_memory | not_retrieved_documents | ONLY_CHAIN_PROMPT | hcx_only | StrOutputParser()
gpt_pipe =  gpt_loaded_memory | not_retrieved_documents | ONLY_CHAIN_PROMPT | gpt_model | StrOutputParser()

gemini_txt_pipe = gemini_loaded_memory | not_retrieved_documents | ONLY_CHAIN_PROMPT | gemini_txt_model | StrOutputParser()
gemini_vis_pipe = RunnablePassthrough() | gemini_vis_model | StrOutputParser()
gemini_vis_txt_pipe = gemini_loaded_memory | img_retrieved_documents | QA_CHAIN_PROMPT | gemini_txt_model | StrOutputParser()
