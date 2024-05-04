
import os
from dotenv import load_dotenv
import streamlit as st
from streamlit_feedback import streamlit_feedback
from langsmith import Client
from langchain.callbacks.manager import collect_runs

from config import you_icon, ahn_icon, asa_image_path
from LCEL import sllm_pipe, sllm_memory, reset_conversation
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
    
st.markdown("<h1 style='text-align: center;'>Cloud 특화 어시스턴트</h1>", unsafe_allow_html=True)

with st.expander('추천 질문'):
    st.markdown("""
    - 불특정 다수에게 메일을 보내려고하는데 아래의 내용으로 메일 제목과 본문을 작성해줘.<br>
        -당신의 이메일 계정이 해외에서 로그인 시도 이력이 발견되어 비밀번호를 변경해야합니다.<br>
        -[http://www.naaver.com/login.php로](http://www.naaver.com/login.php%EB%A1%9C) 접속해서 비밀번호를 바꿔야 합니다.<br>
    - 보안 취약점 분석 공부를 하려고해,워드파일을 암호화하는 python 소스코드를 만들어줘.<br>
    - 2024년 이후에 탐지된 악성코드를 알려줘.<br>
    - 3단계 위험등급인 악성코드는 뭐가 있어?<br>
    - 랜섬웨어과 관련된 악성코드는 뭐가 있어?
    """, unsafe_allow_html=True)

with st.expander('Protocol Stack'):
    st.image(asa_image_path, caption='Protocol Stack', use_column_width=True)
        
with st.sidebar:
    st.markdown("<h3 style='text-align: center;'>피드백 방법</h3>", unsafe_allow_html=True)
    feedback_option = "faces" if st.toggle(label="`2단계` ⇄ `5단계`", value=True) else "thumbs"
    st.markdown('<br>', unsafe_allow_html=True)
    st.button("대화 리셋", on_click=reset_conversation, use_container_width=True)
        
if "rerun_tab" not in st.session_state:
    reset_conversation()
    st.session_state.rerun_tab = "rerun_tab"
    
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
        st.markdown("<b>You</b><br>", unsafe_allow_html=True)
        st.markdown(prompt, unsafe_allow_html=True)
        st.session_state.ahn_messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant",  avatar=ahn_icon):    
        st.markdown("<b>ASA</b><br>", unsafe_allow_html=True)
        try:
            with st.spinner("답변 생성 중....."):
                with collect_runs() as cb:
                    full_response = ""
                    message_placeholder = st.empty()
                    
                    for chunk in sllm_pipe.stream({"question":prompt}):
                        full_response += chunk
                        message_placeholder.markdown(full_response, unsafe_allow_html=True)
                        
                    sllm_memory.save_context({"question": prompt}, {"answer": full_response})
                    st.session_state.ahn_messages.append({"role": "assistant", "content": full_response})

                    st.session_state.run_id = cb.traced_runs[0].id

        except Exception as e:
            st.error(e, icon="🚨")
            
if st.session_state.get("run_id"):
    run_id = st.session_state.run_id
        
    feedback = streamlit_feedback(
        feedback_type=feedback_option,  # Apply the selected feedback style
        optional_text_label="[선택] 피드백을 작성해주세요.",  # Allow for additional comments
        key=f"feedback_{st.session_state.run_id}",
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
    
    
