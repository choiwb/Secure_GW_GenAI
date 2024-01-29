
import streamlit as st
import json
import ssl
from langchain import LLMChain
from typing import Any, List, Optional
from langchain.llms.base import LLM
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain import PromptTemplate
import httpx

# HCX 토큰 계산기 API 호출
from hcx_token_cal import token_completion_executor


API_KEY='API KEY !!!!!!!!!!!!!!!!!!!!!!!!1'
API_KEY_PRIMARY_VAL='API KEY PRIMARY VAL !!!!!!!!!!!!!!!!!!!!!!!!1'
REQUEST_ID='REQUEST ID !!!!!!!!!!!!!!!!!!!!!'
llm_url = 'your llm url !!!!!!!!!!'



SYSTEMPROMPT = """나는 특정 업종 음식점 사장님 이다.

특정 업종 음식점 에 대한 사용자 리뷰의 답변을
사장으로서, 리뷰를 달려고 한다.

<주의 사항>
1. 리뷰 작성 시, 사용자 주문 메뉴를 고려하여, 복수의 메뉴를 주문하였을 때, 
사용자가 리뷰를 1가지 메뉴 만 다는 경우,
다른 주문한 메뉴에 대한 반문 형태의 답변도 포함 되어야 함.
2. 사용자의 리뷰가 긍/부정에 따라, 그에 맞는 이모티콘을 답변 마지막에 반드시 삽입.

<예시>
사용자 주문 메뉴: 간짜장,해물 짬뽕 군만두
사용자 리뷰: 간짜장이 정말 맛있어요.~~~~! 다음번에 또 시켜먹을 계획이에요!
사장님: 간짜장이 정말 맛있었다니 감사합니다. 혹시, 해물 짬뽕과 군만두는 어떠셨는 지요~??^^"""

template = """사용자 주문 메뉴: {menu}
사용자 리뷰: {review}
사장님: """
    
    
    
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context
 


class HCX_stream(LLM):
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

        request_data = {
            'messages': preset_text,
            'topP': 0.8,
            'topK': 0,
            'maxTokens': 512,
            'temperature': 0.9,
            'repeatPenalty': 5.0,
            'stopBefore': [],
            'includeAiFilters': True
        }

        # def execute(self, completion_request):
        headers = {
            'X-NCP-CLOVASTUDIO-API-KEY': API_KEY,
            'X-NCP-APIGW-API-KEY': API_KEY_PRIMARY_VAL,
            'X-NCP-CLOVASTUDIO-REQUEST-ID': REQUEST_ID,
            'Content-Type': 'application/json; charset=utf-8',
            # streaming 옵션 !!!!!
            'Accept': 'text/event-stream'
        }

        full_response = ""
        message_placeholder = st.empty()
        
        with httpx.stream(method="POST", 
                        url=llm_url,
                        json=request_data,
                        headers=headers, 
                        timeout=120) as res:
            for line in res.iter_lines():
                if line.startswith("data:"):
                    split_line = line.split("data:")
                    line_json = json.loads(split_line[1])
                    if "stopReason" in line_json and line_json["stopReason"] == None:
                        full_response += line_json["message"]["content"]
                        print('************************************************************')
                        print(full_response)
                        message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)
            return full_response


hcx_llm = HCX_stream()
prompt = PromptTemplate(template=template, input_variables=["menu", "review"])

hcx_llm_chain = LLMChain(prompt=prompt, llm=hcx_llm)

# if 'generated' not in st.session_state:
#     st.session_state['generated'] = []
 
# if 'past' not in st.session_state:
#     st.session_state['past'] = []
 
# with st.form('form', clear_on_submit=True):
#     # user_input_1 = st.text_input('사용자 주문 메뉴', '', key='menu')
#     # user_input_2 = st.text_input('사용자 리뷰', '', key='review')

#     user_input_1 = st.text_input('사용자 주문 메뉴', default_menu, key='menu')
#     user_input_2 = st.text_input('사용자 리뷰', default_review, key='review')

#     submitted = st.form_submit_button('사장님 답변')
 
#     if submitted and user_input_1 and user_input_2:
#         with st.spinner("Waiting for HyperCLOVA..."): 
#             response_text = hcx_llm_chain.predict(menu = user_input_1, review = user_input_2)

#             single_turn_text_json = {
#             "messages": [
#             {
#                 "role": "system",
#                 "content": template
#             },
#             {
#                 "role": "user",
#                 "content": user_input_1
#             },
#             {
#                 "role": "user",
#                 "content": user_input_2
#             },
#             {
#                 "role": "assistant",
#                 "content": response_text
#             }
#             ]
#             }
            
#             single_turn_text_token = token_completion_executor.execute(single_turn_text_json)
#             print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
#             print(single_turn_text_json)
#             single_turn_token_count = single_turn_text_token[0]['count'] + single_turn_text_token[1]['count'] + single_turn_text_token[2]['count'] + single_turn_text_token[3]['count']
#             single_turn_token_count = sum(token['count'] for token in single_turn_text_token[:4])

#             st.session_state.past.append({'menu': user_input_1, 'review': user_input_2})
#             # st.session_state.generated.append({'generated': response_text, 'token_count': single_turn_token_count})
#             st.session_state.generated.append({'generated': response_text})
 
#         if st.session_state['generated']:
#             for i in range(len(st.session_state['generated']) - 1, -1, -1):
#                 user_input = st.session_state['past'][i]
#                 response = st.session_state['generated'][i]

#                 message(f"사용자 주문 메뉴: {user_input['menu']}", is_user=True, key=str(i) + '_menu')
#                 message(f"사용자 리뷰: {user_input['review']}", is_user=True, key=str(i) + '_review')
#                 message(f"사장님 답변: {response['generated']}", is_user=False, key=str(i) + '_generated')
#                 message(f"총 토큰 수: {response['token_count']}", is_user=False, key=str(i) + '_token_count')




st.title("음식점 사장님 리뷰 자동 생성")
 
default_menu = '양념 치킨 1마리, 치즈볼 5개, 콜라 1.25L'
default_review = '양념 치킨이 존맛탱이고, 다리살이 정말 부드러워용~~ 그리고 가슴살도 안퍽퍽하고 좋아요! 또 시켜먹을게요!'

if 'generated' not in st.session_state:
    st.session_state['generated'] = []
 
if 'past' not in st.session_state:
    st.session_state['past'] = []
    
with st.form('form', clear_on_submit=True):
    # st.subheader("리뷰 입력")
    user_input_2 = st.text_area('사용자 리뷰', default_review, key='review')
    # user_input_1 = st.text_input('사용자 주문 메뉴', default_menu, key='menu')
    st.markdown(f'<div class="small-font">{default_menu}</div>', unsafe_allow_html=True)
    st.session_state['review_width'] = min(max(len(default_menu.split('\n')) * 20, 100), 300)  # 최소 100, 최대 300 픽셀로 조정
    
    submitted = st.form_submit_button('사장님 답변 생성')

    if submitted and user_input_2:

        with st.spinner("사장님의 답변을 생성 중입니다..."):
            response_text = hcx_llm_chain.predict(menu=default_menu, review=user_input_2)
            st.session_state.past.append({'menu': default_menu, 'review': user_input_2, 'response': response_text})
            st.session_state.generated.append({'generated': response_text})
            st.success("생성 완료!")

if st.session_state['past']:
    st.subheader("이전 리뷰 및 사장님 답변")
    for item in reversed(st.session_state['past']):
        st.text_area("리뷰", value=item['review'], disabled=True)
        st.text_area("사장님 답변", value=item['response'], disabled=True)
        st.markdown("---")


