
import pandas as pd
import streamlit as st
from streamlit_hcx_cloud_bot import retrieval_qa_chain, memory

st.title("Cloud 관련 무물보~!")
   
if "messages" not in st.session_state:
    st.session_state.messages = []
            
  
# 저장된 대화 내역과 아바타를 렌더링
for message in st.session_state.messages:
    if message["role"] == "user":
        # 사용자 메시지일 경우, 사용자 아바타 적용
        avatar_icon = message.get("avatar", "🧑")
    else:
        # AI 응답 메시지일 경우, AI 아바타 적용
        avatar_icon = message.get("avatar", "🤖")
    
    with st.chat_message(message["role"], avatar=avatar_icon):
        st.markdown(message["content"])


if prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user", avatar="🧑"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="🤖"):
        message_placeholder = st.empty()        
        full_response = retrieval_qa_chain.invoke({"question":prompt})
        memory.save_context({"question": prompt}, {"answer": full_response})
        
    st.session_state.messages.append({"role": "assistant", "content": full_response})
    
    #  참조 문서 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # total_content = pd.DataFrame(columns=['순번', '참조 문서'])
    # for i in range(len(full_response['source_documents'])):
    #     context = full_response['source_documents'][i].page_content
    #     total_content.loc[i] = [i+1, context]
        
    # st.table(data = total_content)
