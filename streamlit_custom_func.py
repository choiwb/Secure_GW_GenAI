

import time
import json
import streamlit as st


def hcx_stream_process(res):
    full_response = ""
    message_placeholder = st.empty()       
    # 첫 토큰 카운트
    first_token_count = 0
    for line in res.iter_lines():
        if line.startswith("data:"):
            split_line = line.split("data:")
            line_json = json.loads(split_line[1])
            if "stopReason" in line_json and line_json["stopReason"] == None:
                full_response += line_json["message"]["content"]
                # 첫 토큰이 1인 경우, 첫 토큰 생성 시간 측정
                first_token_count += 1
                if first_token_count == 1:
                    end_latency = time.time()
                message_placeholder.markdown(full_response, unsafe_allow_html=True)
            # token length check
            elif "stopReason" in line_json and "seed" in line_json and line_json["stopReason"] == "stop_before":
                token_count = line_json["inputLength"] + line_json["outputLength"] - 1
    return end_latency, token_count, full_response

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
