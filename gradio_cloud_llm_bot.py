
import pandas as pd
import gradio as gr
from streamlit_cloud_llm_bot import retrieval_qa_chain, memory, hcx_llm
from hcx_token_cal import token_completion_executor


src_doc_cols = ['순번', '참조 문서']
def get_completion(user_input):
    response_json = retrieval_qa_chain.invoke({"question": user_input})
    response = response_json['answer']
    memory.save_context({"question": user_input}, {"answer": response})
    
    total_content = pd.DataFrame(columns=src_doc_cols)
    
    for i in range(len(response_json['source_documents'])):
        content = response_json['source_documents'][i].page_content
        total_content.loc[i] = [i+1, content]
        
    output_token_json = {
    "messages": [
    {
        "role": "assistant",
        "content": response
    }
    ]
    }
    
    output_text_token = token_completion_executor.execute(output_token_json)
    output_token_count = sum(token['count'] for token in output_text_token[:])
                
    total_token_count = hcx_llm.total_input_token_count + output_token_count
    
    # 할인 후 가격
    discount_token_price = total_token_count * 0.005
    # 할인 후 가격 VAT 포함
    discount_token_price_vat = discount_token_price * 1.1
    # 정가
    regular_token_price = total_token_count * 0.02
    # 정가 VAT 포함
    regular_token_price_vat = regular_token_price * 1.1    

    token_analysis = f"input 토큰 수: {hcx_llm.total_input_token_count}\noutput 토큰 수: {output_token_count}\n총 토큰 수: {total_token_count}\n할인 후 가격: {round(discount_token_price, 2)} (원)\n할인 후 가격(VAT 포함): {round(discount_token_price_vat, 2)} (원)\n정가: {round(regular_token_price, 2)} (원)\n정가(VAT 포함): {round(regular_token_price_vat, 2)} (원)"
        
    return response, total_content, token_analysis

input_config = gr.Textbox(label="질문")
output_config = [
    gr.Textbox(label="답변"),
    gr.DataFrame(label="참조 문서", headers=src_doc_cols),
    gr.Textbox(label="토큰 분석")
]
gr.close_all()
demo = gr.Interface(fn=get_completion, title = "Cloud 관련 무물보~!", inputs=input_config, outputs=output_config)
demo.launch(share=True)
