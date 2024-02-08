
import pandas as pd
import streamlit as st
from streamlit_cloud_llm_bot import retrieval_qa_chain, memory, cache_instance



st.title("Cloud 관련 무물보~!")
   
if "messages" not in st.session_state:
    st.session_state.messages = []
                            

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
            st.markdown(avatar_message["content"],  unsafe_allow_html=True)

if prompt := st.chat_input("클라우드 컴퓨팅이란 무엇인가요?"):
    with st.chat_message("user", avatar="https://lh3.googleusercontent.com/a/ACg8ocKGr2xjdFlRqAbXU6GCKnYQRDCbttNuDhVJhiLA2Nw8=s432-c-no"):
        st.markdown("<b>You</b><br>" + prompt, unsafe_allow_html=True)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant",  avatar="https://www.shutterstock.com/image-vector/chat-bot-logo-design-concept-600nw-1938811039.jpg"):    
        # HCX_stream 클래스에서 이미 stream 기능을 streamlit ui 에서 구현했으므로 별도의 langchain의 .stream() 필요없고 .invoke()만 호출하면 됨.
        with st.spinner("검색 및 생성 중....."):
            full_response = retrieval_qa_chain.invoke({"question":prompt})               
            # display_message_with_feedback(full_response)
            
            memory.save_context({"question": prompt}, {"answer": full_response})
                    
            # print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
            # print(memory)           
            # memory와는 별도로 cache 된 memory 출력
            # print('&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&')
            # print(cache_instance._cache)

            st.session_state.messages.append({"role": "assistant", "content": full_response})
            
    # 참조 문서 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!                                                                                               
    # total_content = pd.DataFrame(columns=['순번', '참조 문서'])
    # for i in range(len(full_response['source_documents'])):
    #     context = full_response['source_documents'][i].page_content
    #     total_content.loc[i] = [i+1, context]
        
    # st.table(data = total_content)
