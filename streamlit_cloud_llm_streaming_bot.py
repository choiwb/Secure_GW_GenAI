
import streamlit as st
from streamlit_cloud_llm_bot import retrieval_qa_chain, memory

st.title("Cloud 관련 무물보~!")
   
if "messages" not in st.session_state:
    st.session_state.messages = []
  
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
  
if prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="🦖"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="🤖"):
        message_placeholder = st.empty()
        full_response = ""
        # for chunk in retrieval_qa_chain.stream({"question":prompt}):
        for chunk in retrieval_qa_chain.stream(prompt):
            print('-----------------------')
            print(chunk, end="", flush=True)            
            full_response += chunk

        memory.save_context({"question": prompt}, {"answer": full_response})
        print(memory)
       
    st.session_state.messages.append({"role": "assistant", "content": full_response})
