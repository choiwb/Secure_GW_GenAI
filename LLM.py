

import os
import uuid
import random
import json
import httpx
from dotenv import load_dotenv
import requests
import pandas as pd
from typing import Any, List, Optional
from langchain.llms.base import LLM
import streamlit as st
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.chat_models import ChatOpenAI  
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.llms import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

from hcx_token_cal import token_completion_executor
from prompt import PROMPT_INJECTION_PROMPT, SYSTEMPROMPT
from config import sllm_model_path, sllm_n_batch, sllm_n_gpu_layers


##################################################################################
# .env 파일 로드
load_dotenv()

API_KEY=os.getenv('HCX_API_KEY')
API_KEY_PRIMARY_VAL=os.getenv('HCX_API_KEY_PRIMARY_VAL')
REQUEST_ID=str(uuid.uuid4())

os.getenv('OPENAI_API_KEY')

# HCX LLM 경로 !!!!!!!!!!!!!!!!!!!!!!!
llm_url = os.getenv('HCX_LLM_URL')

# Set Google API key
os.getenv("GOOGLE_API_KEY")
os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
##################################################################################



def hcx_stream_process(res):
    full_response = ""
    message_placeholder = st.empty()       
    
    for line in res.iter_lines():
        if line.startswith("data:"):
            split_line = line.split("data:")
            line_json = json.loads(split_line[1])
            if "stopReason" in line_json and line_json["stopReason"] == None:
                full_response += line_json["message"]["content"]
                message_placeholder.markdown(full_response, unsafe_allow_html=True)
    return full_response

class HCX_sec(LLM):        
   
    init_input_token_count: int = 0

    @property
    def _llm_type(self) -> str:
        return "hcx-003"
   
    def _call(
        self,
        prompt: str,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
       
        preset_text = [{"role": "system", "content": PROMPT_INJECTION_PROMPT}, {"role": "user", "content": prompt}]
       
        output_token_json = {
            "messages": preset_text
            }
       
        total_input_token_json = token_completion_executor.execute(output_token_json)
        self.init_input_token_count = sum(token['count'] for token in total_input_token_json[:])
               
        request_data = {
        'messages': preset_text,
        'topP': 0.8,
        'topK': 0,
        'maxTokens': 128,
        'temperature': 0.1,
        'repeatPenalty': 5.0,
        'stopBefore': [],
        'includeAiFilters': True,
        "seed": 4595
        }
       
       
        general_headers = {
            'X-NCP-CLOVASTUDIO-API-KEY': API_KEY,
            'X-NCP-APIGW-API-KEY': API_KEY_PRIMARY_VAL,
            'X-NCP-CLOVASTUDIO-REQUEST-ID': REQUEST_ID,
            'Content-Type': 'application/json; charset=utf-8',
        }
               
        response = requests.post(llm_url, json=request_data, headers=general_headers, verify=False)
        response.raise_for_status()
        
        llm_result = response.json()['result']['message']['content']
        
        preset_text = [{"role": "system", "content": ""}, {"role": "user", "content": llm_result}]
        
        output_token_json = {
            "messages": preset_text
            }
       
        total_input_token_json = token_completion_executor.execute(output_token_json)
        self.init_input_token_count += sum(token['count'] for token in total_input_token_json[:])
                          
        return llm_result
       
 
class HCX_stream(LLM):      
   
    init_input_token_count: int = 0
    source_documents: str = ""
    sample_src_doc_df = pd.DataFrame()

    @property
    def _llm_type(self) -> str:
        return "hcx-003"
   
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
               
        # prompt 변수의 context for answer: 부터 question: 이전 text를 source_documents 선언
        self.source_documents = prompt.split("context for answer: ")[1].split("question: ")[0]
        if len(self.source_documents.strip()) > 0:
            source_documents_list = self.source_documents.split('\n\n')
            sample_src_doc = [[i+1, doc[:100] + '.....(이하 생략)'] for i, doc in enumerate(source_documents_list)] 
            self.sample_src_doc_df = pd.DataFrame(sample_src_doc,  columns=['No', '참조 문서'])
            self.sample_src_doc_df = self.sample_src_doc_df.set_index('No')

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
        'includeAiFilters': True,
        "seed": 4595
        }
                       
        stream_headers = {
            'X-NCP-CLOVASTUDIO-API-KEY': API_KEY,
            'X-NCP-APIGW-API-KEY': API_KEY_PRIMARY_VAL,
            'X-NCP-CLOVASTUDIO-REQUEST-ID': REQUEST_ID,
            'Content-Type': 'application/json; charset=utf-8',
            # streaming 옵션 !!!!!
            'Accept': 'text/event-stream'
        }
        
        with httpx.stream(method="POST",
                        url=llm_url,
                        json=request_data,
                        headers=stream_headers,
                        timeout=10) as res:
            full_response = hcx_stream_process(res)
            return full_response
        
class HCX_only(LLM):        
    
    init_input_token_count: int = 0
 
    @property
    def _llm_type(self) -> str:
        return "hcx-003"
   
    def _call(
        self,
        prompt: str,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
       
        preset_text = [{"role": "system", "content": ""}, {"role": "user", "content": prompt}]
       
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
        'includeAiFilters': True,
        "seed": 4595
        }
                       
        stream_headers = {
            'X-NCP-CLOVASTUDIO-API-KEY': API_KEY,
            'X-NCP-APIGW-API-KEY': API_KEY_PRIMARY_VAL,
            'X-NCP-CLOVASTUDIO-REQUEST-ID': REQUEST_ID,
            'Content-Type': 'application/json; charset=utf-8',
            # streaming 옵션 !!!!!
            'Accept': 'text/event-stream'
        }
        
        with httpx.stream(method="POST",
                        url=llm_url,
                        json=request_data,
                        headers=stream_headers,
                        timeout=10) as res:
            full_response = hcx_stream_process(res)
            return full_response
        
        
gpt_model = ChatOpenAI(
    model="gpt-3.5-turbo",
    # GPT-4 Turbo 
    # model="gpt-4-0125-preview",

    max_tokens=512,
    temperature=0.1
)    

gemini_txt_model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.1, max_output_tokens=512)
gemini_vis_model = ChatGoogleGenerativeAI(model="gemini-pro-vision", temperature=0.1, max_output_tokens=512)



callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

sllm = LlamaCpp(model_path=sllm_model_path, temperature=0, max_tokens=512,
    # context windows
    # n_ctx: 모델이 한 번에 처리할 수 있는 최대 컨텍스트 길이
    n_ctx=8192,
    top_p=1,
    callback_manager=callback_manager, 
    streaming=True, # Streaming is required to pass to the callback manager
    verbose=True, # Verbose is required to pass to the callback manager
    # apple silicon
    f16_kv=True,  # MUST set to True, otherwise you will run into problem after a couple of calls
    n_gpu_layers=sllm_n_gpu_layers,
    n_batch=sllm_n_batch,
    use_mlock=True)
