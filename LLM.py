

import time
from datetime import datetime
import httpx
import requests
from typing import Any, List, Optional
from pydantic import Field
from langchain.llms.base import LLM
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.chat_models import ChatOpenAI  
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.llms import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_aws import ChatBedrock

from prompt import sllm_inj_rag_prompt
from config import HCX_LLM_URL, hcx_003_token_per_price, bedrock_runtime, aws_llm_id, sllm_model_path, sllm_n_batch, sllm_n_gpu_layers, hcx_general_headers, hcx_stream_headers, hcx_llm_params, llm_maxtokens, llm_temperature, gemini_llm_params, gemini_safe, sllm_n_ctx, sllm_top_p
from streamlit_custom_func import hcx_stream_process
from token_usage import record_token_usage

 
class HCX(LLM): 
    question_time: str = ''
    dur_latency: float = 0.0
    init_system_prompt: str
    streaming: bool = Field(default = False)
    token_count: int = 0
    token_price: float = 0.0

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
         
        preset_text = [{"role": "system", "content": self.init_system_prompt}, {"role": "user", "content": prompt}]
        request_data = {
        'messages': preset_text
        }
        total_request_data = request_data | hcx_llm_params
        
        # 질문 요청 시간
        now_time = datetime.now()
        self.question_time = now_time.strftime('%Y%M%d%H%M%S')
        # LLM 요청 시간 
        start_latency = time.time()
        if self.streaming == True:
            with httpx.stream(method="POST",
                            url=HCX_LLM_URL,
                            json=total_request_data,
                            headers=hcx_stream_headers,
                            timeout=10) as res:
                end_latency, self.token_count, full_response = hcx_stream_process(res)
                # 첫 토큰 지연시간
                self.dur_latency = end_latency - start_latency
                self.dur_latency = round(self.dur_latency, 2)
                print('토큰 총 사용량: ', self.token_count)
                self.token_price = self.token_count * hcx_003_token_per_price
                record_token_usage(self.token_count)
                return full_response
            
        else:       
            response = requests.post(HCX_LLM_URL, json=total_request_data, headers=hcx_general_headers, verify=False)
            # 첫 토큰 지연시간
            end_latency = time.time()
            self.dur_latency = end_latency - start_latency
            self.dur_latency = round(self.dur_latency, 2)

            response = response.json()       
            llm_result = response['result']['message']['content']            
            self.token_count = response["result"]["inputLength"] + response["result"]["outputLength"]
            print('토큰 총 사용량: ', self.token_count)
            self.token_price = self.token_count * hcx_003_token_per_price
            record_token_usage(self.token_count)
            return llm_result

gpt_model = ChatOpenAI(
    model="gpt-3.5-turbo",
    # GPT-4 Turbo 
    # model="gpt-4-0125-preview",

    max_tokens=llm_maxtokens,
    temperature=llm_temperature
)    

gemini_txt_model = ChatGoogleGenerativeAI(model="gemini-pro", gemini_confog=gemini_llm_params,
                                          convert_system_message_to_human=True, safety_settings=gemini_safe)
gemini_vis_model = ChatGoogleGenerativeAI(model="gemini-pro-vision", gemini_confog=gemini_llm_params, 
                                          convert_system_message_to_human=True, safety_settings=gemini_safe)


# Claude Model configuration
aws_model_kwargs =  { 
    "max_tokens": llm_maxtokens, "temperature": llm_temperature,
}

# LangChain class for chat - Claude
sonnet_llm = ChatBedrock(
    client=bedrock_runtime,
    model_id=aws_llm_id,
    model_kwargs=aws_model_kwargs
)

callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

sllm = LlamaCpp(model_path=sllm_model_path, temperature=llm_temperature, max_tokens=llm_maxtokens,
    # context windows
    # n_ctx: 모델이 한 번에 처리할 수 있는 최대 컨텍스트 길이
    n_ctx=sllm_n_ctx,
    top_p=sllm_top_p,
    callback_manager=callback_manager, 
    streaming=True, # Streaming is required to pass to the callback manager
    verbose=True, # Verbose is required to pass to the callback manager
    # apple silicon
    f16_kv=True,  # MUST set to True, otherwise you will run into problem after a couple of calls
    n_gpu_layers=sllm_n_gpu_layers,
    n_batch=sllm_n_batch,
    use_mlock=True,
    prompt=sllm_inj_rag_prompt)
