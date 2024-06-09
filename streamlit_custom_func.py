

import time
import json
import pandas as pd
import streamlit as st

from config import compression_retriever, user_compression_retriever


def hcx_stream_process(res):
    full_response = ""
    message_placeholder = st.empty()       
    
    for line in res.iter_lines():
        if line.startswith("data:"):
            split_line = line.split("data:")
            line_json = json.loads(split_line[1])
            if "stopReason" in line_json and line_json["stopReason"] == None:
                full_response += line_json["message"]["content"]
                message_placeholder.markdown(full_response, unsafe_allow_html=True)
    return full_response

def scroll_bottom():
    js = f"""
    <script>
        // 스크롤을 하단으로 이동시키는 함수
        function scrollToBottom(){{
            var textAreas = parent.document.querySelectorAll('section.main');
            for (let index = 0; index < textAreas.length; index++) {{
                textAreas[index].scrollTop = textAreas[index].scrollHeight;
            }}
        }}

        // MutationObserver의 콜백 함수 정의
        function observeMutations(){{
            var observer = new MutationObserver(scrollToBottom);
            var config = {{ childList: true, subtree: true }};
            // 감시 대상 요소 지정 및 옵저버 시작
            var target = parent.document.querySelector('section.main');
            if(target) observer.observe(target, config);
        }}

        // 초기 스크롤 위치 조정 및 DOM 변화 감지를 위한 옵저버 설정
        scrollToBottom();
        observeMutations();
    </script>
    """
    st.components.v1.html(js, height=0)
    
def src_doc(prompt):
    prompt_str = str(prompt)
    source_documents = prompt_str.split("context for answer: ")[1].split("question: ")[0]
    extraction_question = prompt_str.split("question: ")[1]
    
    if len(source_documents.strip()) > 0 and source_documents.strip() != '\\n':
        source_documents_list = source_documents.split('\\n\\n')
        sample_src_doc = [[i+1, doc[:100] + '.....(이하 생략)'] for i, doc in enumerate(source_documents_list)] 
        sample_src_doc_df = pd.DataFrame(sample_src_doc,  columns=['No', '참조 문서'])
        sample_src_doc_df = sample_src_doc_df.set_index('No')

        # 문서 추출 소요 시간: 3.16 (초) => 불필요하므로 제거하는것도 고민해 봐야 함!!!!!!
        start = time.time()
        if st.session_state.selected_db == 'user_vectordb':
            extraction_context = user_compression_retriever.invoke(extraction_question)
        else:
            extraction_context = compression_retriever.invoke(extraction_question)
        end = time.time()
        dur = end - start
        print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
        print('문서 추출 소요 시간: %.2f (초)' %(dur))
        print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
        
        src_doc_score = [doc.state['query_similarity_score'] for doc in extraction_context]
        src_doc_locat = [doc.metadata for doc in extraction_context]
        
        sample_src_doc_df['유사도'] = src_doc_score
        sample_src_doc_df['유사도'] = sample_src_doc_df['유사도'].round(2).astype(str)
        sample_src_doc_df['문서 출처'] = src_doc_locat
                
        # 참조 문서 UI 표출
        if sample_src_doc_df.shape[0] > 0:
            with st.expander('참조 문서'):
                st.table(sample_src_doc_df)
                st.markdown("AhnLab에서 제공하는 위협정보 입니다.<br>자세한 정보는 https://www.ahnlab.com/ko/contents/asec/info 에서 참조해주세요.", unsafe_allow_html=True)
    return prompt
