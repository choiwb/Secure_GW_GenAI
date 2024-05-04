
import os
from dotenv import load_dotenv
import streamlit as st
from PIL import Image
from langchain.schema import HumanMessage
from streamlit_feedback import streamlit_feedback
from langsmith import Client
from langchain.callbacks.manager import collect_runs

from config import asa_image_path, you_icon, ahn_icon
from prompt import gemini_img_sys_message
from LCEL import reset_conversation, gemini_memory, gemini_vis_pipe, gemini_vis_vectordb_txt_pipe
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
    st.markdown("<h3 style='text-align: center;'>이미지 업로드</h3>", unsafe_allow_html=True)
    uploaded_image = st.file_uploader("이미지 선택", type=["png", "jpg", "jpeg"])

    if uploaded_image is not None:        
        # 업로드된 이미지를 보여주기 위해 Image 객체로 변환
        image = Image.open(uploaded_image)
        
        # 사이드바에 이미지 표시
        st.image(image, caption="업로드된 이미지", use_column_width=True)
        
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
                    st.session_state.image_data = uploaded_image
                                            
                    gemini_img_message = HumanMessage(
                        content=[
                            {
                            "type": "text",
                            "text": "Provide information of given image."},
                            {"type": "image_url", "image_url": image},
                        ]
                        )
                    img_context = gemini_vis_pipe.invoke([gemini_img_sys_message, gemini_img_message])
                    
                    for chunk in gemini_vis_vectordb_txt_pipe.stream({"img_context":img_context, "question":prompt}):
                        full_response += chunk
                        message_placeholder.markdown(full_response, unsafe_allow_html=True)   
                        
                    gemini_memory.save_context({"question": prompt}, {"answer": full_response})
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
