
import pandas as pd
import streamlit as st
from streamlit_cloud_llm_bot import retrieval_qa_chain, memory

st.title("Cloud 관련 무물보~!")
   
# 메시지와 좋아요/싫어요 버튼을 함께 표시.
# def display_message_with_feedback(message):
#     # st.text_area("답변", value=message, height=100, disabled=True)
#     col1, col2 = st.columns([1, 1])
#     with col1:
#         if st.button("👍", key="like"):
#             st.write("감사합니다! 피드백이 반영되었습니다.")
#             # 좋아요 피드백을 처리하는 로직을 여기에 추가
#     with col2:
#         if st.button("👎", key="dislike"):
#             st.write("피드백을 주셔서 감사합니다. 개선할 수 있도록 노력하겠습니다.")
#             # 싫어요 피드백을 처리하는 로직을 여기에 추가

if "messages" not in st.session_state:
    st.session_state.messages = []
    
# if 'generated' not in st.session_state:
#     st.session_state['generated'] = []

# if 'past' not in st.session_state:
#     st.session_state['past'] = []
                          
# 저장된 대화 내역과 아바타를 렌더링
for avatar_message in st.session_state.messages:
    if avatar_message["role"] == "user":
        # 사용자 메시지일 경우, 사용자 아바타 적용
        avatar_icon = avatar_message.get("avatar", "🧑")
    else:
        # AI 응답 메시지일 경우, AI 아바타 적용
        avatar_icon = avatar_message.get("avatar", "🤖")

    with st.chat_message(avatar_message["role"], avatar=avatar_icon):
        st.markdown(avatar_message["content"])

if prompt := st.chat_input("클라우드 컴퓨팅이란 무엇인가요?"):
    with st.chat_message("user", avatar="🧑"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant",  avatar="🤖"):    
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



    
# if prompt := st.chat_input("클라우드 컴퓨팅이란 무엇인가요?"):
    
#     with st.chat_message("user", avatar="🧑"):
#         st.markdown(prompt)   
        
    # with st.chat_message("assistant", avatar="🤖"):
    #     with st.spinner("검색 및 생성 중....."):
    #         response = retrieval_qa_chain.invoke({"question":prompt})    
    #         memory.save_context({"question": prompt}, {"answer": response})    

#     # store the output 
#     st.session_state.past.append(prompt)
#     st.session_state.generated.append(response)

# if st.session_state['generated']:   
#     # for i in range(len(st.session_state['generated'])-1, -1, -1):
#     for i in range(len(st.session_state['generated'])):
#         message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')
#         message(st.session_state["generated"][i], key=str(i))
