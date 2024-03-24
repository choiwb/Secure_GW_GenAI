

import os
import streamlit as st
from PIL import Image
from streamlit_cloud_llm_bot import reset_conversation, gemini_memory, gemini_txt_pipe, gemini_vis_pipe
from langchain.schema import HumanMessage

########################################################################
you_icon = os.path.join(os.getcwd(), 'image/you_icon.png')
ahn_icon = os.path.join(os.getcwd(), 'image/ahn_icon.png')

# asa, hcx 별 프로토콜 스택 이미지 경로
asa_image_path = os.path.join(os.getcwd(), 'image/protocol_stack.png')
########################################################################


try:
    st.set_page_config(layout="wide")
except Exception as e:
    # 페이지를 자동으로 다시 실행
    st.rerun()
    
    
st.markdown("<h1 style='text-align: center;'>Visual Question Answering Assistant</h1>", unsafe_allow_html=True)

with st.expander('Protocol Stack'):
    st.image(asa_image_path, caption='Protocol Stack', use_column_width=True)

if 'image_data' not in st.session_state:
    st.session_state.image_data = None
    
if "ahn_messages" not in st.session_state:
    st.session_state.ahn_messages = []


for avatar_message in st.session_state.ahn_messages:
    if avatar_message["role"] == "user":
        # 사용자 메시지일 경우, 사용자 아바타 적용
        avatar_icon = avatar_message.get("avatar", you_icon)
        with st.chat_message(avatar_message["role"], avatar=avatar_icon):
            st.markdown("<b>You</b><br>" + avatar_message["content"], unsafe_allow_html=True)
    else:
        # AI 응답 메시지일 경우, AI 아바타 적용
        avatar_icon = avatar_message.get("avatar", ahn_icon)
        with st.chat_message(avatar_message["role"], avatar=avatar_icon):
            with st.expander('ASA'):
                st.markdown("<b>ASA</b><br>" + avatar_message["content"], unsafe_allow_html=True)

        
with st.sidebar:
    st.button("대화 리셋", on_click=reset_conversation(), use_container_width=True)

    uploaded_image = st.file_uploader("이미지 업로드", type=["png", "jpg", "jpeg"])
    if uploaded_image is not None:
        st.session_state.image_data = uploaded_image
        
        # 업로드된 이미지를 보여주기 위해 Image 객체로 변환
        image = Image.open(uploaded_image)
        
        # 사이드바에 이미지 표시
        st.image(image, caption="업로드된 이미지", use_column_width=True)
    

if prompt := st.chat_input(""):
    with st.chat_message("user", avatar=you_icon):
        st.markdown("<b>You</b><br>" + prompt, unsafe_allow_html=True)
        st.session_state.ahn_messages.append({"role": "user", "content": prompt})
        
    with st.chat_message("assistant",  avatar=ahn_icon):    
        try:     
            with st.spinner("검색 및 생성 중....."):
                    # uploaded_image = st.session_state.image_data
                    
                    full_response = "<b>ASA</b><br>"
                    message_placeholder = st.empty()
                    
                    if uploaded_image is None:
                        for chunk in gemini_txt_pipe.stream({"question":prompt}):
                                    print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
                                    print(chunk)
                                    full_response += chunk
                                    message_placeholder.markdown(full_response + "▌", unsafe_allow_html=True)
                        message_placeholder.markdown(full_response, unsafe_allow_html=True)   
                    else:
                        st.session_state.image_data = uploaded_image
                        image = Image.open(uploaded_image)
                        
                        img_message = HumanMessage(
                        content=[
                            {
                            "type": "text",
                            "text": "Provide information on menu and price of given image."},
                            {"type": "image_url", "image_url": image},
                        ]
                        )
                                                                   
                        for chunk in gemini_vis_pipe.stream([img_message]):
                                    print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
                                    print(uploaded_image)
                                    print(chunk)
                                    full_response += chunk
                                    message_placeholder.markdown(full_response + "▌", unsafe_allow_html=True)
                        message_placeholder.markdown(full_response, unsafe_allow_html=True)   
                        
                    full_response_for_token_cal = full_response.replace('<b>Assistant</b><br>', '').replace('<b>ASA</b><br>', '')
                    gemini_memory.save_context({"question": prompt}, {"answer": full_response_for_token_cal})

                    print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                    print(gemini_memory)
                    print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')

                    st.session_state.ahn_messages.append({"role": "assistant", "content": full_response_for_token_cal})
                    
        except Exception as e:
            st.error(e, icon="🚨")

