
import os
import json
import httpx
import time
import requests
import ssl
from pydantic import Extra, Field
from typing import Any, List, Mapping, Optional
from langchain.prompts import PromptTemplate
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS, Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain_community.document_loaders import PyPDFLoader
from langchain.llms.base import LLM
import streamlit as st
from streamlit_chat import message
from langchain.callbacks.manager import CallbackManagerForLLMRun
import pandas as pd
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from operator import itemgetter
from langchain_core.messages import get_buffer_string
from langchain.schema import format_document
from langchain.retrievers import ContextualCompressionRetriever
from langchain.document_loaders import AsyncHtmlLoader
from langchain.document_transformers import Html2TextTransformer
from langchain.chains import create_extraction_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chat_models import ChatOpenAI  
from langchain.retrievers.document_compressors import EmbeddingsFilter
from multiprocessing import Pool
import concurrent.futures

# HCX 토큰 계산기 API 호출
from hcx_token_cal import token_completion_executor


##################################################################################
# HCX API 키
API_KEY='YOUR API KEY !!!!!!!!!!!!!!!!!!!!!!!='
API_KEY_PRIMARY_VAL='YOUR API KEY PRIMARY VAL !!!!!!!!!!!!!!!!!!!!!!!'
REQUEST_ID='YOUR REQUEST ID !!!!!!!!!!!!!!!!!!!!!!!'

# (개인) 유료 API 키!!!!!!!!
os.environ['OPENAI_API_KEY'] = "YOUR OPENAI API KEY !!!!!!!!!!!!!!!!!!!!!!!"

# 임베딩 벡터 DB 저장 & 호출
db_save_path = "YOUR DB SAVE PATH !!!!!!!!!!!!!!!!!!!!!!!" 

# HCX LLM 경로 !!!!!!!!!!!!!!!!!!!!!!!
llm_url = 'your llm url !!!!!!!!!!'

# pdf 형태 context 경로 !!!!!!!!
pdf_path_1 = 'your pdf context rag data path !!!!!!!!!!!!!!!!!!'
pdf_path_2 = 'your pdf context rag data path !!!!!!!!!!!!!!!!!!'

ahn_asec_path = 'pdf files dir !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!'
##################################################################################

pdf_paths = []
for filename in os.listdir(ahn_asec_path):
    if filename.endswith(".pdf"):
        # 완전한 파일 경로 생성
        globals()[f'pdf_path_{filename}']  = os.path.join(ahn_asec_path, filename)
        # print(globals()[f'pdf_path_{filename}'])
        pdf_paths.append(globals()[f'pdf_path_{filename}'])

SYSTEMPROMPT = """You are a Cloud (MSP) Engineer or Cloud Sales administrator, or Cloud Solution Architect. about user question, answering specifically in korean.
    Use the following pieces of context to answer the question at the end.
    You mast answer after understanding previous conversation.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    Respond don't know to questions not related to Cloud Computing.
    Use 5 sentences maximum and keep the answer as concise as possible."""
template = """
    context for answer: {context}
    question: {question}
    answer: """
    
QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context", "question"],template=template)

text_splitter = CharacterTextSplitter(        
                            separator = "\n",
                            chunk_size = 200,
                            chunk_overlap  = 50,
                            length_function = len,
                            )

text_splitter_function_calling = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=200, chunk_overlap=50
    )


# text-embedding-3-small or text-embedding-3-large
embeddings = OpenAIEmbeddings(model = 'text-embedding-3-small')


try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

class CompletionExecutor(LLM):
    api_key: str = Field(...)
    api_key_primary_val: str = Field(...)
    request_id: str = Field(...)
    system_prompt: str = Field(...)

    # RAG 과정 시 2번 씩 프린트 되는 flow 여서, init, total 별도 선언 해줘야 함.
    total_input_token_count: int = 0
    
    class Config:
        extra = Extra.forbid
 
    def __init__(self, api_key, api_key_primary_val, request_id, system_prompt):

        super().__init__()
        self.api_key = api_key
        self.api_key_primary_val = api_key_primary_val
        self.request_id = request_id
        self.system_prompt = system_prompt

 
    @property
    def _llm_type(self) -> str:
        return "HyperClovaX"
 
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,        
    ) -> str:
        if stop is not None:
            raise ValueError("stop kwargs are not permitted.")
 
        headers = {
            'X-NCP-CLOVASTUDIO-API-KEY': self.api_key,
            'X-NCP-APIGW-API-KEY': self.api_key_primary_val,
            'X-NCP-CLOVASTUDIO-REQUEST-ID': self.request_id,
            'Content-Type': 'application/json; charset=utf-8'
        }

        # self.system_prompt 의 경우, 처음에 1번만 이용 !!!!!!!!!!!!!
        # if self.total_input_token_count == 0:
        #     # preset_text = [{"role": "system", "content": self.system_prompt}, {"role": "user", "content": prompt}]
        #     preset_text = [{"role": "system", "content": self.system_prompt}]            
        # else:
        #     preset_text = [{"role": "user", "content": prompt}]
            
        preset_text = [{"role": "system", "content": self.system_prompt}, {"role": "user", "content": prompt}]
                        
        output_token_json = {
            "messages": preset_text
            }
        
        total_input_token_json = token_completion_executor.execute(output_token_json)
        init_input_token_count = sum(token['count'] for token in total_input_token_json[:])
        
        # RAG 과정 시 2번 씩 프린트 되는 flow 여서, init 을 total 에 합침
        self.total_input_token_count += init_input_token_count

        
        payload = {
            'messages': preset_text,
            'topP': 0.8,
            'topK': 0,
            'maxTokens': 256,
            'temperature': 0.1,
            'repeatPenalty': 5.0,
            'stopBefore': [],
            'includeAiFilters': True
        }

        response = requests.post(self.llm_url, json=payload, headers=headers, verify=False)
        response.raise_for_status()

        return response.json()['result']['message']['content']
 
    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {"llmUrl": self.llm_url}

hcx_llm = CompletionExecutor(api_key = API_KEY, api_key_primary_val=API_KEY_PRIMARY_VAL, request_id=REQUEST_ID,  system_prompt=SYSTEMPROMPT)




class HCX_general(LLM):        
    
    init_input_token_count: int = 0

    @property
    def _llm_type(self) -> str:
        return "HyperClovaX"
    
    def _call(
        self,
        prompt: str,
        # stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        # if stop is not None:
        #    raise ValueError("stop kwargs are not permitted.")
        
        preset_text = [{"role": "system", "content": SYSTEMPROMPT}, {"role": "user", "content": prompt}]

        # print('---------------------------------------------')
        # print(preset_text)
        
        output_token_json = {
            "messages": preset_text
            }
        
        total_input_token_json = token_completion_executor.execute(output_token_json)
        self.init_input_token_count = sum(token['count'] for token in total_input_token_json[:])
                
        request_data = {
        'messages': preset_text,
        'topP': 0.8,
        'topK': 0,
        'maxTokens': 256,
        'temperature': 0.1,
        'repeatPenalty': 5.0,
        'stopBefore': [],
        'includeAiFilters': True
        }
        
        
        general_headers = {
            'X-NCP-CLOVASTUDIO-API-KEY': API_KEY,
            'X-NCP-APIGW-API-KEY': API_KEY_PRIMARY_VAL,
            'X-NCP-CLOVASTUDIO-REQUEST-ID': REQUEST_ID,
            'Content-Type': 'application/json; charset=utf-8',
        }
                
        response = requests.post(llm_url, json=request_data, headers=general_headers, verify=False)
        response.raise_for_status()
        print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
        print(response)
                    
        return response.json()['result']['message']['content']
        


class HCX_stream(LLM):      
    
    init_input_token_count: int = 0
    source_documents: str = ""
    
    @property
    def _llm_type(self) -> str:
        return "HyperClovaX"
    
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        if stop is not None:
            raise ValueError("stop kwargs are not permitted.")
        
        preset_text = [{"role": "system", "content": SYSTEMPROMPT}, {"role": "user", "content": prompt}]

        print('---------------------------------------------')
        print(preset_text)
        
        # print('*************************************************')
        # prompt 변수의 context for answer: 부터 question: 이전 text를 source_documents 선언
        self.source_documents = prompt.split("context for answer: ")[1].split("question: ")[0]
        # print(source_documents)
        
        output_token_json = {
            "messages": preset_text
            }
        
        total_input_token_json = token_completion_executor.execute(output_token_json)
        self.init_input_token_count = sum(token['count'] for token in total_input_token_json[:])
        
        request_data = {
        'messages': preset_text,
        'topP': 0.8,
        'topK': 0,
        'maxTokens': 512,
        'temperature': 0.1,
        'repeatPenalty': 5.0,
        'stopBefore': [],
        'includeAiFilters': True
        }
                        
        stream_headers = {
            'X-NCP-CLOVASTUDIO-API-KEY': API_KEY,
            'X-NCP-APIGW-API-KEY': API_KEY_PRIMARY_VAL,
            'X-NCP-CLOVASTUDIO-REQUEST-ID': REQUEST_ID,
            'Content-Type': 'application/json; charset=utf-8',
            # streaming 옵션 !!!!!
            'Accept': 'text/event-stream'
        }

        
        # streaming 형태로 최종 출력 도출
        # full_response = ""
        full_response = "<b>Assistant</b><br>"

        message_placeholder = st.empty()
        
        with httpx.stream(method="POST", 
                        url=llm_url,
                        json=request_data,
                        headers=stream_headers, 
                        timeout=120) as res:
            for line in res.iter_lines():
                if line.startswith("data:"):
                    split_line = line.split("data:")
                    line_json = json.loads(split_line[1])
                    if "stopReason" in line_json and line_json["stopReason"] == None:
                        full_response += line_json["message"]["content"]
                        print('************************************************************')
                        print(full_response)
                        message_placeholder.markdown(full_response + "▌", unsafe_allow_html=True)
            message_placeholder.markdown(full_response, unsafe_allow_html=True)
            
            return full_response



hcx_general = HCX_general()
hcx_stream = HCX_stream()



# 오프라인 데이터 가공 ####################################################################################
def offline_faiss_save(*pdf_path):

    total_docs = []
    
    for pdf_url in pdf_path:
        pdfreader =  PyPDFLoader(pdf_url)
        pdf_doc = pdfreader.load_and_split()
        doc = text_splitter.split_documents(pdf_doc)
        print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        print(len(doc))
        total_docs = total_docs + doc
        
    print('@@@@@@@@@@@@@@@@@@@@@@@@@@')
    print(len(total_docs))

    # Convert list of dictionaries to strings
    total_content = [str(item) for item in total_docs]

    docsearch = FAISS.from_texts(total_content, embeddings)

    docsearch.embedding_function
    docsearch.save_local(os.path.join(db_save_path, "cloud_bot_20240208_faiss_db"))


# start = time.time()
# total_content = offline_faiss_save(pdf_path_1, pdf_path_2)
# end = time.time()
# '''임베딩 완료 시간: 1.62 (초)'''
# print('임베딩 완료 시간: %.2f (초)' %(end-start))

def offline_chroma_save(pdf_paths):

    total_docs = []
    
    for pdf_url in pdf_paths:
        pdfreader =  PyPDFLoader(pdf_url)
        pdf_doc = pdfreader.load_and_split()
        doc = text_splitter.split_documents(pdf_doc)
        print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        print(len(doc))
        total_docs = total_docs + doc
        
    print('@@@@@@@@@@@@@@@@@@@@@@@@@@')
    print(len(total_docs))
    # Convert list of dictionaries to strings
    # total_content = [str(item) for item in total_docs]    
    
    vectorstore = Chroma.from_documents(
        documents=total_docs, 
        embedding=embeddings,
        persist_directory=os.path.join(db_save_path, "cloud_bot_20240214_chroma_db")
        )
    vectorstore.persist()

# start = time.time()
# total_content = offline_chroma_save(pdf_paths)
# end = time.time()
# '''임베딩 완료 시간: 1.31 (초)'''
# print('임베딩 완료 시간: %.2f (초)' %(end-start))
#######################################################################################################



# 온라인 데이터 가공 ####################################################################################
# 비정형 데이터 => 정형 데이터 가공 (도표 추출 등) 위한 Function Calling 구현 필요 !!!!!!!!!!!!!!!!!!!!!
# url_0 = 'https://cloud.google.com/docs/get-started/aws-azure-gcp-service-comparison?hl=ko'
# url_0 = 'https://m.ahnlab.com/ko/contents/asec/info/37285'
url_0 = 'https://www.ncloud.com/product/compute/gpuServer'
url_1 = 'https://www.ncloud.com/product/networking/loadBalancer'

# Function Calling
# url_0_schema = {
#     "properties": {
#         "서비스 카테고리": {"type": "string"},
#         "서비스 유형": {"type": "string"},
#         "Google Cloud 제품": {"type": "string"},
#         "Google Cloud 제품 설명": {"type": "string"},
#         "AWS 제공": {"type": "string"},
#         "Azure 제공": {"type": "string"}
#     },
#     "required": ["서비스 카테고리", "서비스 유형", "Google Cloud 제품", "Google Cloud 제품 설명", "AWS 제공", "Azure 제공"],
# }



# # from langchain.chat_models import ChatOpenAI
# openai_llm = ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0, max_tokens=2048,
#                 streaming=False)

# # function calling - HCX
# hcx_prep = HCX_general() | StrOutputParser()

# def extract(content: str, schema: dict):
#     extracted_content = create_extraction_chain(schema=schema, llm=hcx_prep).invoke(content)
#     # extracted_content = create_extraction_chain(schema=schema, llm=openai_llm).invoke(content)

#     return extracted_content

# def extract_content(schema, content):
#     # 토큰의 길이를 확인하고, 4096을 초과하지 않으면 내용 추출
#     if len(content) <= 4096:
#         return extract(schema=schema, content=content)
    
#     # 토큰의 길이가 4096을 초과하면, 내용을 절반으로 나누고 각 부분에 대해 재귀적으로 처리
#     half = len(content) // 2
#     first_half_content = extract_content(schema, content[:half])
#     second_half_content = extract_content(schema, content[half:])
    
#     return first_half_content + second_half_content



# def online_multiple_prep(args):
#     schema, page_content = args
#     try:
#         extracted_content = extract_content(schema=schema, content=page_content)
#         extracted_content = extracted_content['text']
#         return extracted_content
#     except Exception as e:
#         # Handle the exception as needed
#         print(e)
#         # openai 의 분당 최대 토큰 수 초과 관련 에러 발생할 경우, 아래 코드 적용 !!!!!!!!!!
#         time.sleep(60)


# html2text = Html2TextTransformer()

# def online_chroma_save(url):
#     total_docs = []
    
#     loader = AsyncHtmlLoader(url)        
#     doc = loader.load()
#     doc = html2text.transform_documents(doc)  
    
#     # print("Extracting content with LLM")
#     splits  = text_splitter_function_calling.split_documents(doc)
#     print('################################################################')
#     print(len(splits))
        
#     with concurrent.futures.ThreadPoolExecutor() as executor:
#         # Prepare arguments for extract_content_wrapper
#         args_list = [(url_0_schema, split.page_content) for split in splits]

#         # Process each split concurrently
#         results = list(executor.map(online_multiple_prep, args_list))

#         # Filter out None values (from exceptions in extract_content_wrapper)
#         total_docs.extend(filter(None, results))
            
#     total_docs = [str(item) for item in total_docs]
#     print('111111111111111111111111111111111111')
#     print(len(total_docs))
    
#     # vectorstore = Chroma.from_documents(
#     #     documents=total_docs, 
#     #     embedding=embeddings,
#     #     persist_directory=os.path.join(db_save_path, "cloud_bot_20240119_chroma_db")
#     #     )
#     vectorstore = Chroma.from_texts(
#         texts=total_docs, 
#         embedding=embeddings,
#         persist_directory=os.path.join(db_save_path, "cloud_bot_20240131_chroma_db")
#         )
#     vectorstore.persist()
    
    
# if __name__ == "__main__":
#     start = time.time()
#     # total_content = online_chroma_save(url_0)
#     cpu_num = int(os.cpu_count() / 2)
#     with Pool(processes=cpu_num) as pool:
#         total_content = pool.map(online_chroma_save, [url_0])
#     end = time.time()
#     '''임베딩 완료 시간: 168.88 (초)'''
#     print('임베딩 완료 시간: %.2f (초)' %(end-start))
#######################################################################################################




# 온라인 데이터 가공 ####################################################################################
# Funcation Calling 사용 X

def online_chroma_save(*urls):
    
    html2text = Html2TextTransformer()
    total_splits = []
    
    for url in urls:
        loader = AsyncHtmlLoader(url)        
        doc = loader.load()
        doc = html2text.transform_documents(doc)  
        
        splits = text_splitter_function_calling.split_documents(doc)
        print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        print(len(splits))
        # multiple url 에 대한 splits 를 append !!!!!!!
        total_splits = total_splits + splits
        
    print('@@@@@@@@@@@@@@@@@@@@@@@@@')
    print(len(total_splits))
    
    vectorstore = Chroma.from_documents(
        documents=splits, 
        embedding=embeddings,
        persist_directory=os.path.join(db_save_path, "cloud_bot_20240209_chroma_db")
        )
    vectorstore.persist()

# start = time.time()
# total_content = online_chroma_save(url_0, url_1)
# end = time.time()
# '''임베딩 완료 시간: 1.62 (초)'''
# print('임베딩 완료 시간: %.2f (초)' %(end-start))

new_docsearch = Chroma(persist_directory=os.path.join(db_save_path, "cloud_bot_20240214_chroma_db"),
                        embedding_function=embeddings)

retriever = new_docsearch.as_retriever(
                                        search_type="mmr",                                        
                                        search_kwargs={'k': 8, 'fetch_k': 32}
                                       )

# # retriever의 compression 시도 !!!!!!!!!!!!!!!!!!!!!!!!!
embeddings_filter = EmbeddingsFilter(embeddings=embeddings, similarity_threshold=0.5)

compression_retriever = ContextualCompressionRetriever(
    base_compressor=embeddings_filter, base_retriever=retriever
)


# langchain 기반 memory caching
from langchain.cache import InMemoryCache
from langchain.globals import set_llm_cache

cache_instance = InMemoryCache()
set_llm_cache(cache_instance)

@st.cache_resource
def init_memory():
    return ConversationBufferMemory(
        input_key='question',
        output_key='answer',
        memory_key='chat_history',
        return_messages=True)

memory = init_memory()

# 토큰 절약하기 위한
# ConversationalRetrievalChain to LCEL !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
DEFAULT_DOCUMENT_PROMPT = PromptTemplate.from_template(template="{page_content}")

_template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""
CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)

def _combine_documents(
    docs, document_prompt=DEFAULT_DOCUMENT_PROMPT, document_separator="\n\n"
):
    doc_strings = [format_document(doc, document_prompt) for doc in docs]
    return document_separator.join(doc_strings)

# First we add a step to load memory
# This adds a "memory" key to the input object
loaded_memory = RunnablePassthrough.assign(
    chat_history=RunnableLambda(memory.load_memory_variables) | itemgetter("chat_history"),
)

# # Now we calculate the standalone question
standalone_question = {
    "standalone_question": {
        "question": lambda x: x["question"],
        "chat_history": lambda x: get_buffer_string(x["chat_history"]),
    }
    | CONDENSE_QUESTION_PROMPT
    | hcx_general
    | StrOutputParser()
}

# Now we retrieve the documents
retrieved_documents = {
    "source_documents": itemgetter("standalone_question") | retriever,
    # "source_documents": itemgetter("standalone_question") | compression_retriever,
    "question": lambda x: x["standalone_question"],
}

# Now we construct the inputs for the final prompt
final_inputs = {
    "context": lambda x: _combine_documents(x["source_documents"]),
    "question": itemgetter("question"),
}

# And finally, we do the part that returns the answers 
answer = {
    "answer": final_inputs | QA_CHAIN_PROMPT | hcx_stream,
    "source_documents": itemgetter("source_documents"),
}

# stream 기능이 있는 llm 클래스의 경우, 위 lcel의 answer 처럼 파이프라인 안에서 선언하면 안되고, 아래 코드와 같이 별도로 선언해야 함 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# 따라서 stream으로 최종 출력을 뽑를 경우, 위 lcel의 answer 과정의 source_documents 를 추출 못하여 참조 문서를 표출 못하는거 같음.....
# retrieval_qa_chain = loaded_memory | standalone_question | retrieved_documents | answer
retrieval_qa_chain = loaded_memory | standalone_question | retrieved_documents | final_inputs | QA_CHAIN_PROMPT | hcx_stream | StrOutputParser()



# st.title("Cloud 관련 무물보~!")

# if 'generated' not in st.session_state:
#     st.session_state['generated'] = []  
 
# if 'past' not in st.session_state:
#     st.session_state['past'] = []
        
# with st.form('form', clear_on_submit=True):
#     user_input = st.text_input('질문', '', key='ques')

#     submitted = st.form_submit_button('답변')
        
#     if submitted and user_input:
#         with st.spinner("Waiting for HyperCLOVA..."):   
#             # LCEL     
#             response_text_json = retrieval_qa_chain.invoke({"question": user_input})
                        
#             # ConversationalRetrievalChain & LCEL      
#             response_text = response_text_json['answer']
                        
#             # Note that the memory does not save automatically
#             # This will be improved in the future
#             # For now you need to save it yourself
#             memory.save_context({"question": user_input}, {"answer": response_text})
#             # print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
#             # print(memory)

#             # 참조 문서 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#             total_content = pd.DataFrame(columns=['순번', '참조 문서'])
#             token_total_content = ''
#             for i in range(len(response_text_json['source_documents'])):
#                 context = response_text_json['source_documents'][i].page_content
#                 # print('==================================================')
#                 # print('\n%d번 째 참조 문서: %s' %(i+1, context))
#                 # total_content += '=================================================='
#                 # total_content += '\n%d번 째 참조 문서: %s' %(i+1, context)
#                 total_content.loc[i] = [i+1, context]
#                 token_total_content += context
                                
#             output_token_json = {
#             "messages": [
#             {
#                 "role": "assistant",
#                 "content": response_text
#             }
#             ]
#             }
            
#             output_text_token = token_completion_executor.execute(output_token_json)
#             output_token_count = sum(token['count'] for token in output_text_token[:])
                        
#             total_token_count = hcx_llm.total_input_token_count + output_token_count
            
#             # 할인 후 가격
#             discount_token_price = total_token_count * 0.005
#             # 할인 후 가격 VAT 포함
#             discount_token_price_vat = discount_token_price * 1.1
#             # 정가
#             regular_token_price = total_token_count * 0.02
#             # 정가 VAT 포함
#             regular_token_price_vat = regular_token_price * 1.1

#             st.session_state.past.append({'question': user_input})
#             st.session_state.generated.append({'generated': response_text, 'input_token_count':hcx_llm.total_input_token_count,
#                                                'output_token_count': output_token_count,
#                                                'total_token_count': total_token_count,
#                                               'discount_token_price': discount_token_price,
#                                               'discount_token_price_vat': discount_token_price_vat,
#                                               'regular_token_price': regular_token_price,
#                                               'regular_token_price_vat': regular_token_price_vat}
#                                               )
            
#             response_container = st.empty()  # Use an empty container to update the response dynamically
#             full_response = ""
#             for chunk in response_text:
#                 full_response += chunk
#                 # Update the response dynamically within the same container
#                 # response_container.markdown(f"답변: {full_response}")
#                 message(f"답변: {full_response}", is_user=False)
#                 time.sleep(0.01)                
        
#         if st.session_state['generated']:
#             for i in range(len(st.session_state['generated']) - 1, -1, -1):
#                 user_input = st.session_state['past'][i]
#                 response = st.session_state['generated'][i]

#                 message(f"질문: {user_input['question']}", is_user=True, key=str(i) + '_question')
#                 message(f"답변: {response['generated']}", is_user=False, key=str(i) + '_generated')
#                 message(f"input 토큰 수: {response['input_token_count']}\noutput 토큰 수: {response['output_token_count']}\n총 토큰 수: {response['total_token_count']}\n할인 후 가격: {round(response['discount_token_price'], 2)} (원)\n할인 후 가격(VAT 포함): {round(response['discount_token_price_vat'], 2)} (원)\n정가: {round(response['regular_token_price'], 2)} (원)\n정가(VAT 포함): {round(response['regular_token_price_vat'], 2)} (원)", is_user=False, key=str(i) + '_cost')
      
#             st.table(data = total_content)



################### Streamlit ###################
# streamlit run streamlit_cloud_llm_bot.py
