
import os
import pandas as pd
import streamlit as st
from streamlit_cloud_llm_bot import retrieval_qa_chain, memory, cache_instance, hcx_general, hcx_stream
from streamlit_feedback import streamlit_feedback
from langsmith import Client
from langchain.callbacks.manager import collect_runs

# HCX 토큰 계산기 API 호출
from hcx_token_cal import token_completion_executor

##################################################################################
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = f"your langsmith project name !!!!!!!!!!!!!!!!!!!"
os.environ["LANGCHAIN_ENDPOINT"] = 'https://api.smith.langchain.com'
os.environ["LANGCHAIN_API_KEY"] = 'your langsmith api key !!!!!!!!!!!!!!!!!!!!'
##################################################################################

client = Client()

    
st.title("Cloud 특화 챗봇")
      
if "messages" not in st.session_state:
    st.session_state.messages = []
                            
if st.sidebar.button("Clear message history"):
    print("Clearing message history")
    memory.clear()
    st.session_state.trace_link = None
    st.session_state.run_id = None
    
# 저장된 대화 내역과 아바타를 렌더링
for avatar_message in st.session_state.messages:
    if avatar_message["role"] == "user":
        # 사용자 메시지일 경우, 사용자 아바타 적용
        avatar_icon = avatar_message.get("avatar", "https://lh3.googleusercontent.com/a/ACg8ocKGr2xjdFlRqAbXU6GCKnYQRDCbttNuDhVJhiLA2Nw8=s432-c-no")
        with st.chat_message(avatar_message["role"], avatar=avatar_icon):
            st.markdown("<b>You</b><br>" + avatar_message["content"], unsafe_allow_html=True)
    else:
        # AI 응답 메시지일 경우, AI 아바타 적용
        avatar_icon = avatar_message.get("avatar", "https://www.shutterstock.com/image-vector/chat-bot-logo-design-concept-600nw-1938811039.jpg")
        with st.chat_message(avatar_message["role"], avatar=avatar_icon):
            # HCX_stream 클래스에서 "Assistant" 를 이미 bold 처리하여 생성하므로, 굳이 더할 필요는 없음! 하지만 unsafe_allow_html = True를 해야 함.
            st.markdown("<b>Assistant</b><br>" + avatar_message["content"],  unsafe_allow_html=True)

if prompt := st.chat_input(""):
    with st.chat_message("user", avatar="https://lh3.googleusercontent.com/a/ACg8ocKGr2xjdFlRqAbXU6GCKnYQRDCbttNuDhVJhiLA2Nw8=s432-c-no"):
        st.markdown("<b>You</b><br>" + prompt, unsafe_allow_html=True)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant",  avatar="https://www.shutterstock.com/image-vector/chat-bot-logo-design-concept-600nw-1938811039.jpg"):    
        # HCX_stream 클래스에서 이미 stream 기능을 streamlit ui 에서 구현했으므로 별도의 langchain의 .stream() 필요없고 .invoke()만 호출하면 됨.        
        with st.spinner("검색 및 생성 중....."):
            with collect_runs() as cb:
                full_response = retrieval_qa_chain.invoke({"question":prompt})               
                
                # full_response에서 <b>Assistant</b><br> 제거
                full_response_for_token_cal = full_response.replace('<b>Assistant</b><br>', '')
                output_token_json = {
                "messages": [
                {
                    "role": "assistant",
                    "content": full_response_for_token_cal
                }
                ]
                }

                memory.save_context({"question": prompt}, {"answer": full_response_for_token_cal})
                
                # print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                # print(memory)           
                # # memory와는 별도로 cache 된 memory 출력
                # print('&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&')
                # print(cache_instance._cache)
                
                st.session_state.messages.append({"role": "assistant", "content": full_response_for_token_cal})

                st.session_state.run_id = cb.traced_runs[0].id
                
                output_text_token = token_completion_executor.execute(output_token_json)
                output_token_count = sum(token['count'] for token in output_text_token[:])

                total_token_count = hcx_general.init_input_token_count + hcx_stream.init_input_token_count + output_token_count

                # 총 토큰의 경우 langchain이 아닌 NCP의 입력 및 출력 토큰 별도 적용 !!!!!!!!!!!!!
                hcx_input_token_count = hcx_general.init_input_token_count + hcx_stream.init_input_token_count
                hcx_output_token_count = output_token_count
                hcx_total_token_count = total_token_count
            
                st.markdown(f"입력 토큰 수: {hcx_input_token_count}")
                st.markdown(f"출력 토큰 수: {hcx_output_token_count}")
                st.markdown(f"총 토큰 수: {hcx_total_token_count}")
                
                one_dashboard_log = list(client.list_runs(
                    project_name='Cloud Chatbot - Monitoring 20240210',
                    run_type="llm",
                    start_time=cb.traced_runs[0].start_time,
                ))
                
                # print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
                # 총 길이: 2
                # print(len(one_dashboard_log))
                
                # print(one_dashboard_log[0].prompt_tokens)
                # print(one_dashboard_log[0].completion_tokens)
                # print(one_dashboard_log[0].total_tokens)
                
                # print(one_dashboard_log[1].prompt_tokens)
                # print(one_dashboard_log[1].completion_tokens)
                # print(one_dashboard_log[1].total_tokens)

                # one_dashboard_log[0].prompt_tokens = hcx_input_token_count
                # one_dashboard_log[0].completion_tokens = hcx_output_token_count
                # one_dashboard_log[0].total_tokens = hcx_total_token_count
                # one_dashboard_log[1].prompt_tokens = hcx_input_token_count
                # one_dashboard_log[1].completion_tokens = hcx_output_token_count
                # one_dashboard_log[1].total_tokens = hcx_total_token_count
                # print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
                # print(one_dashboard_log[0].prompt_tokens)
                # print(one_dashboard_log[0].completion_tokens)
                # print(one_dashboard_log[0].total_tokens)
                
                # Assuming `client` is an instance of LangSmith's Client and one_dashboard_log[0] has an ID
                # updated_log = {
                #     "prompt_tokens": hcx_input_token_count,
                #     "completion_tokens": hcx_output_token_count,
                #     "total_tokens": hcx_total_token_count
                # }

                # Hypothetical method to update a run's details
                # client.update_run(run_id=st.session_state.run_id, update_data=updated_log)

    # 참조 문서 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!                                                                                               
    total_content = pd.DataFrame(columns=['참조 문서'])
    total_content.loc[0] = [hcx_stream.source_documents]
    st.table(data = total_content)
    
    
feedback_option = "faces" if st.toggle(label="`Thumbs` ⇄ `Faces`", value=False) else "thumbs"

if st.session_state.get("run_id"):
    run_id = st.session_state.run_id
    feedback = streamlit_feedback(
        feedback_type=feedback_option,  # Apply the selected feedback style
        optional_text_label="[Optional] Please provide an explanation",  # Allow for additional comments
        key=f"feedback_{st.session_state.run_id}",
    )
    
    updated_log = {
            "prompt_tokens": hcx_input_token_count,
            "completion_tokens": hcx_output_token_count,
            "total_tokens": hcx_total_token_count
        }
    # print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
    # print(updated_log)
    # Hypothetical method toㄴ update a run's details
    client.update_run(run_id=run_id, total_tokens=hcx_total_token_count)

    # Define score mappings for both "thumbs" and "faces" feedback systems
    score_mappings = {
        "thumbs": {"👍": 1, "👎": 0},
        "faces": {"😀": 1, "🙂": 0.75, "😐": 0.5, "🙁": 0.25, "😞": 0},
    }

    # Get the score mapping based on the selected feedback option
    scores = score_mappings[feedback_option]

    if feedback:
        # Get the score from the selected feedback option's score mapping
        score = scores.get(feedback["score"])

        if score is not None:
            # Formulate feedback type string incorporating the feedback option
            # and score value
            feedback_type_str = f"{feedback_option} {feedback['score']}"

            # Record the feedback with the formulated feedback type string
            # and optional comment
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
            st.toast("Feedback recorded!", icon="📝")
        else:
            st.warning("Invalid feedback score.")
