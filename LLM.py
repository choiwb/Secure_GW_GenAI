

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

from hcx_token_cal import token_CompletionExecutor
from prompt import sllm_inj_rag_prompt
from config import HCX_LLM_URL, bedrock_runtime, aws_llm_id, sllm_model_path, sllm_n_batch, sllm_n_gpu_layers, hcx_general_headers, hcx_stream_headers, hcx_llm_params, llm_maxtokens, llm_temperature, gemini_llm_params, gemini_safe, sllm_n_ctx, sllm_top_p
from streamlit_custom_func import hcx_stream_process
from token_usage import record_token_usage

token_completion_executor = token_CompletionExecutor()
 
class HCX(LLM): 
    # init_input_token_count: int = 0
    init_system_prompt: str
    streaming: bool = Field(default = False)

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

        # total_input_token_json = token_completion_executor.execute(request_data)
        # self.init_input_token_count = sum(token['count'] for token in total_input_token_json[:])

        if self.streaming == True:
            with httpx.stream(method="POST",
                            url=HCX_LLM_URL,
                            json=total_request_data,
                            headers=hcx_stream_headers,
                            timeout=10) as res:
                total_count, full_response = hcx_stream_process(res)
                print('토큰 총 사용량: ', total_count)
                return full_response
            
        else:       
            response = requests.post(HCX_LLM_URL, json=total_request_data, headers=hcx_general_headers, verify=False)
            # response.raise_for_status()     
            response = response.json()       
            llm_result = response['result']['message']['content']
            # preset_text = [{"role": "system", "content": ""}, {"role": "user", "content": llm_result}]
            # output_token_json = {
            #     "messages": preset_text
            #     }
            # total_input_token_json = token_completion_executor.execute(output_token_json)
            # self.init_input_token_count += sum(token['count'] for token in total_input_token_json[:])
            
            total_count = response["result"]["inputLength"] + response["result"]["outputLength"]
            print('토큰 총 사용량: ', total_count)
            record_token_usage(total_count)
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


