
import pandas as pd
from langchain.prompts import PromptTemplate
from langchain.schema import SystemMessage, HumanMessage
from langchain.memory import ConversationBufferMemory
import streamlit as st
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.runnables import chain
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from operator import itemgetter
from langchain_core.messages import get_buffer_string
from langchain.schema import format_document

from config import compression_retriever, user_compression_retriever
from prompt import not_rag_template, rag_template, img_rag_template, PROMPT_INJECTION_PROMPT, SYSTEMPROMPT
from LLM import HCX, gpt_model, sonnet_llm, sllm, gemini_vis_model, gemini_txt_model


ONLY_CHAIN_PROMPT = PromptTemplate(input_variables=["question"],template=not_rag_template)
QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context", "question"],template=rag_template)
IMG_QA_CHAIN_PROMPT = PromptTemplate(input_variables=["img_context", "valid_img_context", "question"],template=img_rag_template)
 
class doc_review_system_key(BaseModel):
    문서명: str = Field(description="문서명을 추출해줘.")
    조합명: str = Field(description="조합명(단체명)을 추출해줘, 단 추출 시, 띄어쓰기를 반드시 주의해야 해.")
    고유번호: str = Field(description="고유번호를 추출해줘, 숫자 3자리 - 숫자 2자리 - 숫자 5자리 형태야.")
    주소: str = Field(description="주소(소재지)를 추출해줘.")
    대표자성명: str = Field(description="대표자 성명을 추출해줘.")
    생년월일: str = Field(description="생년월일을 추출해줘.")
    조합인감: str = Field(description="조합인감의 유/무를 O/X로 나타내줘.")

json_parser = JsonOutputParser(pydantic_object=doc_review_system_key)

def multimodal_prompt_json_parser(img_base64):        
    return [
    SystemMessage(
    content=""""Answer the user question.\n{format_instructions}\n{question}\n"""
    ),
    HumanMessage(
            content=[
                {
                    "type": "image_url",
                    "image_url": {
                    "url": f"data:image/png;base64,{img_base64}", 
                    },
                },
                {
                    "type": "text", "text": json_parser.get_format_instructions()
                }
            ]
    )
    ]

hcx_sec = HCX(init_system_prompt = PROMPT_INJECTION_PROMPT, streaming = False)
hcx_stream = HCX(init_system_prompt = SYSTEMPROMPT, streaming = True)

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
    
# def src_doc(init_src_doc):
#     if len(init_src_doc['source_documents']) > 0:
#         source_documents_total = init_src_doc['source_documents']
        
#         src_doc_context = [doc.page_content for doc in source_documents_total]
#         src_doc_score = [doc.state['query_similarity_score'] for doc in source_documents_total]
#         src_doc_metadata = [doc.metadata for doc in source_documents_total]

#         formatted_metadata = [
#             f'문서 명: {metadata["source"].split("/")[-1]}, 문서 위치: {metadata["page"]} 쪽'
#             for metadata in src_doc_metadata
#         ]
        
#         src_doc_df = pd.DataFrame({'참조 문서': src_doc_context, '유사도': src_doc_score, '문서 출처': formatted_metadata})
#         src_doc_df['No'] = [i+1 for i in range(src_doc_df.shape[0])]
#         src_doc_df = src_doc_df.set_index('No')
#         src_doc_df['참조 문서'] = src_doc_df['참조 문서'].str.slice(0, 100) + '.....(이하 생략)'
#         src_doc_df['유사도'] = src_doc_df['유사도'].round(3).astype(str)
        
#         with st.expander('참조 문서'):
#             st.table(src_doc_df)
#             st.markdown("AhnLab에서 제공하는 위협정보 입니다.<br>자세한 정보는 https://www.ahnlab.com/ko/contents/asec/info 에서 참조해주세요.", unsafe_allow_html=True)

#     return init_src_doc

class SrcDoc():
    def src_doc(init_src_doc):
        if len(init_src_doc['source_documents']) > 0:
            source_documents_total = init_src_doc['source_documents']
            
            src_doc_context = [doc.page_content for doc in source_documents_total]
            src_doc_score = [doc.state['query_similarity_score'] for doc in source_documents_total]
            src_doc_metadata = [doc.metadata for doc in source_documents_total]

            formatted_metadata = [
                f'문서 명: {metadata["source"].split("/")[-1]}, 문서 위치: {metadata["page"]} 쪽'
                for metadata in src_doc_metadata
            ]
            print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
            print(formatted_metadata)
            
            src_doc_df = pd.DataFrame({'참조 문서': src_doc_context, '유사도': src_doc_score, '문서 출처': formatted_metadata})
            src_doc_df['No'] = [i+1 for i in range(src_doc_df.shape[0])]
            src_doc_df = src_doc_df.set_index('No')
            src_doc_df['참조 문서'] = src_doc_df['참조 문서'].str.slice(0, 100) + '.....(이하 생략)'
            src_doc_df['유사도'] = src_doc_df['유사도'].round(3).astype(str)
            
            with st.expander('참조 문서'):
                st.table(src_doc_df)
                st.markdown("AhnLab에서 제공하는 위협정보 입니다.<br>자세한 정보는 https://www.ahnlab.com/ko/contents/asec/info 에서 참조해주세요.", unsafe_allow_html=True)

        return init_src_doc

not_retrieved_documents = {
    "question": lambda x: x["question"],
    "chat_history": lambda x: get_buffer_string(x["chat_history"])
}

img_retrieved_documents = {
    "question": lambda x: x["question"],
    "img_context": lambda x: x["img_context"],
    "valid_img_context": lambda x: x["valid_img_context"],
    "chat_history": lambda x: get_buffer_string(x["chat_history"])
}
 
final_inputs = {
    "context": lambda x: _combine_documents(x["source_documents"]),
    "question": itemgetter("question"),
    "chat_history": itemgetter("chat_history")
    }
 
img_final_inputs = {
    "img_context": itemgetter("img_context"),
    "valid_img_context": itemgetter("valid_img_context"),
    "question": itemgetter("question"),
    "chat_history": itemgetter("chat_history")
}

hcx_sec_pipe = ONLY_CHAIN_PROMPT | hcx_sec | StrOutputParser()
retrieval_qa_chain = asa_loaded_memory | retrieved_documents | SrcDoc.src_doc | final_inputs | QA_CHAIN_PROMPT | hcx_stream | StrOutputParser()
user_retrieval_qa_chain = asa_loaded_memory | user_retrieved_documents | SrcDoc.src_doc | final_inputs | QA_CHAIN_PROMPT | hcx_stream | StrOutputParser()hcx_only_pipe =  hcx_loaded_memory | not_retrieved_documents |  ONLY_CHAIN_PROMPT | hcx_stream | StrOutputParser()
gpt_pipe =  gpt_loaded_memory | not_retrieved_documents | ONLY_CHAIN_PROMPT | gpt_model | StrOutputParser()
aws_retrieval_qa_chain = asa_loaded_memory | retrieved_documents | src_doc | final_inputs | QA_CHAIN_PROMPT | sonnet_llm | StrOutputParser()
sllm_pipe = sllm_loaded_memory | retrieved_documents | src_doc | final_inputs | QA_CHAIN_PROMPT | sllm | StrOutputParser()

gemini_txt_pipe = gemini_loaded_memory | not_retrieved_documents | ONLY_CHAIN_PROMPT | gemini_txt_model | StrOutputParser()
gemini_vis_pipe = RunnablePassthrough() | gemini_vis_model | StrOutputParser()
gemini_vis_txt_pipe = (
                                gemini_loaded_memory | img_retrieved_documents | img_final_inputs | 
                                IMG_QA_CHAIN_PROMPT | gemini_txt_model | StrOutputParser()
                               )
aws_vis_json_parser_pipe = RunnablePassthrough() | sonnet_llm | json_parser
aws_vis_txt_pipe = (
                                asa_loaded_memory | img_retrieved_documents | img_final_inputs | 
                                IMG_QA_CHAIN_PROMPT | sonnet_llm | StrOutputParser()
                               )
