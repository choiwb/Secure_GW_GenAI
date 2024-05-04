

import streamlit as st
from PIL import Image
from langchain.schema import HumanMessage

from config import asa_image_path, you_icon, ahn_icon
from LCEL import reset_conversation, gemini_memory, gemini_vis_pipe, gemini_vis_txt_pipe


try:
    # st.set_page_config(page_icon="🚀", page_title="Cloud_Assistant", layout="wide", initial_sidebar_state="collapsed")
    st.set_page_config(page_icon="🚀", page_title="Cloud_Assistant", layout="wide")
except:
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
            st.markdown("<b>You</b><br>", unsafe_allow_html=True)
            st.markdown(avatar_message["content"], unsafe_allow_html=True)
    else:
        # AI 응답 메시지일 경우, AI 아바타 적용
        avatar_icon = avatar_message.get("avatar", ahn_icon)
        with st.chat_message(avatar_message["role"], avatar=avatar_icon):
            with st.expander('ASA'):
                st.markdown("<b>ASA</b><br>", unsafe_allow_html=True)
                st.markdown(avatar_message["content"], unsafe_allow_html=True)
                
with st.sidebar:
    st.markdown("<h3 style='text-align: center;'>이미지 업로드</h3>", unsafe_allow_html=True)
    uploaded_image = st.file_uploader("", type=["png", "jpg", "jpeg"])

    if uploaded_image is not None:        
        # 업로드된 이미지를 보여주기 위해 Image 객체로 변환
        image = Image.open(uploaded_image)
        
        # 사이드바에 이미지 표시
        st.image(image, caption="업로드된 이미지", use_column_width=True)
        
    st.markdown('<br>', unsafe_allow_html=True)
    st.button("대화 리셋", on_click=reset_conversation, use_container_width=True)
        
if prompt := st.chat_input(""):
    with st.chat_message("user", avatar=you_icon):
        st.markdown("<b>You</b><br>" + prompt, unsafe_allow_html=True)
        st.session_state.ahn_messages.append({"role": "user", "content": prompt})
        
    with st.chat_message("assistant",  avatar=ahn_icon):    
        try:     
            with st.spinner("답변 생성 중....."):
                full_response = "<b>ASA</b><br>"
                message_placeholder = st.empty()
                
                st.session_state.image_data = uploaded_image
                
                img_message = HumanMessage(
                content=[
                    {
                    "type": "text",
                    "text": "Provide information of given image."},
                    {"type": "image_url", "image_url": image},
                ]
                )
                                                            
                img_context = gemini_vis_pipe.invoke([img_message])
                                                
                for chunk in gemini_vis_txt_pipe.stream({"context":img_context, "question":prompt}):
                    full_response += chunk
                    message_placeholder.markdown(full_response, unsafe_allow_html=True)   
                    
                full_response_for_token_cal = full_response.replace('<b>Assistant</b><br>', '').replace('<b>ASA</b><br>', '')
                gemini_memory.save_context({"question": prompt}, {"answer": full_response_for_token_cal})

                st.session_state.ahn_messages.append({"role": "assistant", "content": full_response_for_token_cal})
                    
        except Exception as e:
            st.error(e, icon="🚨")

