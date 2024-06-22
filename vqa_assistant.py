
import os
import base64
from io import BytesIO
from dotenv import load_dotenv
import streamlit as st
from streamlit_feedback import streamlit_feedback
from langsmith import Client
from langchain.callbacks.manager import collect_runs
from langchain_community.document_loaders import PyPDFLoader
from pdf2image import convert_from_bytes

from config import asa_image_path, you_icon, ahn_icon, user_pdf_folder_path
from prompt import multimodal_prompt
from LCEL import (reset_conversation, gemini_memory, asa_memory, gemini_vis_pipe, aws_vis_pipe, 
                  gemini_vis_vectordb_txt_pipe, aws_vis_vectordb_txt_pipe)
from streamlit_custom_func import scroll_bottom

##################################################################################
# .env 파일 로드
load_dotenv()

os.getenv('LANGCHAIN_TRACING_V2')
os.getenv('LANGCHAIN_PROJECT')
os.getenv('LANGCHAIN_ENDPOINT')
os.getenv('LANGCHAIN_API_KEY')
##################################################################################

client = Client()

try:
    # st.set_page_config(page_icon="🚀", page_title="Cloud_Assistant", layout="wide", initial_sidebar_state="collapsed")
    st.set_page_config(page_icon="🚀", page_title="Cloud_Assistant", layout="wide")
except:
    st.rerun()
    
st.markdown("<h1 style='text-align: center;'>Visual Question Answering Assistant</h1>", unsafe_allow_html=True)

with st.expander('Protocol Stack'):
    st.image(asa_image_path, caption='Protocol Stack', use_column_width=True)
    
with st.sidebar:
    st.markdown("<h3 style='text-align: center;'>pdf 업로드</h3>", unsafe_allow_html=True)
    uploaded_pdf = st.file_uploader("pdf 선택", type=["pdf"])

    if uploaded_pdf is not None:        
        # 멀티모달 사용 (문서에서 서명, 인감 등 추출) 위해 pdf to image 
        pages = convert_from_bytes(uploaded_pdf.read())
        
        images_base64 = []
        for page_number, image in enumerate(pages):
            st.markdown(f"### Page {page_number + 1}")
            st.image(image, caption=f"업로드된 이미지 - 페이지 {page_number + 1}", use_column_width=True)
                    
            width, height = image.size 
            # print(f"width: {width}, height: {height}, size: {width*height}")

            ######################################################################        
            # # 이미지를 2 x 2 로 나누었을 때 오른쪽 하단 부분 추출
            # left = width // 2
            # top = height // 2
            # right = width
            # bottom = height
            
            # cropped_image = image.crop((left, top, right, bottom))
            
            # # 추출한 이미지 표시
            # st.image(cropped_image, caption="추출된 이미지 (오른쪽 하단)", use_column_width=True)
            # width, height = cropped_image.size 
            ######################################################################
                    
            isResized = False
            # 5242880: Claude 3 Sonnet 의 허용 사이즈
            while(width*height > 5242880):                    
                width = int(width/2)
                height = int(height/2)
                isResized = True
                # print(f"width: {width}, height: {height}, size: {width*height}")
                        
            if isResized:
                img = image.resize((width, height))
                # cropped_image = cropped_image.resize((width, height))
                                
            buffer = BytesIO()
            image.save(buffer, format="PNG")
            # cropped_image.save(buffer, format="PNG")

            img_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
            images_base64.append(img_base64)
            
    st.markdown('<br>', unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center;'검증용 pdf 업로드</h3>", unsafe_allow_html=True)
    valid_uploaded_pdf = st.file_uploader("검증용 pdf 선택", type=["pdf"])

    if valid_uploaded_pdf is not None:        
        # 멀티모달 사용 (문서에서 서명, 인감 등 추출) 위해 pdf to image 
        pages = convert_from_bytes(valid_uploaded_pdf.read())
        
        valid_images_base64 = []
        for page_number, image in enumerate(pages):
            st.markdown(f"### Page {page_number + 1}")
            st.image(image, caption=f"업로드된 이미지 - 페이지 {page_number + 1}", use_column_width=True)
                    
            width, height = image.size 
            # print(f"width: {width}, height: {height}, size: {width*height}")
                    
            isResized = False
            # 5242880: Claude 3 Sonnet 의 허용 사이즈
            while(width*height > 5242880):                    
                width = int(width/2)
                height = int(height/2)
                isResized = True
                # print(f"width: {width}, height: {height}, size: {width*height}")
                        
            if isResized:
                img = image.resize((width, height))
                                
            buffer = BytesIO()
            image.save(buffer, format="PNG")

            valid_img_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
            valid_images_base64.append(valid_img_base64)
            
    st.markdown('<br>', unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center;'>피드백 방법</h3>", unsafe_allow_html=True)
    feedback_option = "faces" if st.toggle(label="`2단계` ⇄ `5단계`", value=True) else "thumbs"
    st.markdown('<br>', unsafe_allow_html=True)
    st.button("대화 리셋", on_click=reset_conversation, use_container_width=True)
    
if "rerun_tab" not in st.session_state:
    reset_conversation()
    st.session_state.rerun_tab = "rerun_tab"
    
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
                        
if prompt := st.chat_input(""):
    scroll_bottom()    
    with st.chat_message("user", avatar=you_icon):
        st.markdown("<b>You</b><br>" + prompt, unsafe_allow_html=True)
        st.session_state.ahn_messages.append({"role": "user", "content": prompt})
        
    with st.chat_message("assistant",  avatar=ahn_icon):    
        st.markdown("<b>ASA</b><br>", unsafe_allow_html=True)
        try:     
            with st.spinner("답변 생성 중....."):
                with collect_runs() as cb:
                    full_response = ""
                    message_placeholder = st.empty()
                    st.session_state.image_data = uploaded_pdf

                    img_context_total = []
                    for i in images_base64:    
                        img_context = aws_vis_pipe.invoke(multimodal_prompt(i)) 
                        img_context_total.append(img_context)
                    img_context_total = '\n'.join(img_context_total)

                    valid_img_context_total = []
                    for i in valid_images_base64:    
                        valid_img_context = aws_vis_pipe.invoke(multimodal_prompt(i)) 
                        valid_img_context_total.append(valid_img_context)
                    valid_img_context_total = '\n'.join(valid_img_context_total)
                                        
                    for chunk in aws_vis_vectordb_txt_pipe.stream({"img_context":img_context_total, "valid_img_context": valid_img_context_total, "question":prompt}):
                        full_response += chunk
                        message_placeholder.markdown(full_response, unsafe_allow_html=True)   
                        
                    asa_memory.save_context({"question": prompt}, {"answer": full_response})
                    st.session_state.ahn_messages.append({"role": "assistant", "content": full_response})
                                                
                    # multimodal llm 결과에 대한 피드백은 필요 없음!
                    multimodal_llm_run_id = cb.traced_runs[0].id
                    # 사용자 피드백이 필요한 질문에 대한 결과 !!
                    st.session_state.run_id = cb.traced_runs[1].id
                    
        except Exception as e:
            st.error(e, icon="🚨")

if st.session_state.get("run_id"):
    run_id = st.session_state.run_id
        
    feedback = streamlit_feedback(
        feedback_type=feedback_option,  # Apply the selected feedback style
        optional_text_label="[선택] 피드백을 작성해주세요.",  # Allow for additional comments
        key=f"feedback_{run_id}"
    )

    score_mappings = {
        "thumbs": {"👍": 1, "👎": 0},
        "faces": {"😀": 1, "🙂": 0.75, "😐": 0.5, "🙁": 0.25, "😞": 0},
    }

    scores = score_mappings[feedback_option]

    if feedback:
        score = scores.get(feedback["score"])

        if score is not None:
            feedback_type_str = f"{feedback_option} {feedback['score']}"

            feedback_record = client.create_feedback(
                run_id,
                feedback_type_str,
                score=score,
                comment=feedback.get("text")
                )
        
            st.session_state.feedback = {
                "feedback_id": str(feedback_record.id),
                "score": score,
            }
            st.toast("피드백 등록!", icon="📝")
        else:
            st.warning("부적절한 피드백.")
