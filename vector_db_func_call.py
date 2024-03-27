
import os
from langchain.chains import create_extraction_chain
from multiprocessing import Pool
from langchain.chat_models import ChatOpenAI
from streamlit_cloud_llm_bot import pdf_paths, embeddings, db_save_path, text_splitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.vectorstores import FAISS


scraping_llm = ChatOpenAI(model_name='gpt-3.5-turbo-16k', temperature=0, max_tokens=8192)

raw_to_json_schema = {
  "title": "악성코드 요약",
  "description": "악성코드의 특징들을 추출하는 전처리",
  "type": "object",
  "properties": {
    "악성코드 이름": {
      "type": "string",
      "title": "악성코드 이름",
      "description": "악성코드 이름"
    },
    "악성코드 탐지일": {
      "type": "string",
      "title": "악성코드 탐지일",
      "description": "악성코드 탐지일"
    },
    
    "위험 정보": {
      "type": "object",
      "title": "위험 정보",
      "description": "악성코드의 위험 정보",
      "properties": {
        "시스템 위험도": {
          "type": "string",
          "title": "시스템 위험도",
          "description": "악성코드의 위험 정보의 시스템 위험도"
        },
        "네트워크 위험도": {
          "type": "string",
          "title": "네트워크 위험도",
          "description": "악성코드의 위험 정보의 네트워크 위험도"
        },
        "확산 위험도": {
          "type": "string",
          "title": "확산 위험도",
          "description": "악성코드의 위험 정보의 확산 위험도"
        },
        "위험 등급": {
          "type": "string",
          "title": "위험 등급",
          "description": "악성코드의 위험 정보의 위험 등급"
        },
      }
    },
    
    "기본 정보": {
      "type": "object",
      "title": "기본 정보",
      "description": "악성코드의 기본 정보",
      "properties": {
        "종류": {
          "type": "string",
          "title": "종류",
          "description": "악성코드의 기본 정보의 종류"
        },
        "운영 체제": {
          "type": "string",
          "title": "운영 체제",
          "description": "악성코드의 기본 정보의 운영 체제"
        },
        "유입 경로": {
          "type": "string",
          "title": "유입 경로",
          "description": "악성코드의 기본 정보의 유입 경로"
        },
        "공격 기법": {
          "type": "string",
          "title": "공격 기법",
          "description": "악성코드의 기본 정보의 공격 기법"
        },
        "영향도": {
          "type": "string",
          "title": "영향도",
          "description": "악성코드의 기본 정보의 영향도"
        },
      }
    },
    
      "증상 및 요약": {
      "type": "string",
      "title": "증상 및 요약",
      "description": "악성코드 증상 및 요약"
    },
      
      "상세 정보": {
      "type": "object",
      "title": "상세 정보",
      "description": "악성코드의 상세 정보",
      "properties": {
        "전파 경로": {
          "type": "string",
          "title": "전파 경로",
          "description": "악성코드의 상세 정보의 전파 경로"
        },
        "실행 후 증상": {
          "type": "string",
          "title": "실행 후 증상",
          "description": "악성코드의 상세 정보의 실행 후 증상"
        }
      }
    },
      
    "파일 생성": {
      "type": "string",
      "title": "파일 생성",
      "description": "악성코드의 파일 생성"
    },
          
    "프로세스 생성": {
      "type": "string",
      "title": "프로세스 생성",
      "description": "악성코드의 프로세스 생성"
    },
    
    "레지스트리 설정": {
      "type": "string",
      "title": "레지스트리 설정",
      "description": "악성코드의 레지스트리 설정"
    },
    
    "C&C 서버": {
      "type": "string",
      "title": "C&C 서버",
      "description": "악성코드의 C&C 서버"
    },
}
}

# def offline_faiss_save(pdf_paths):
 
#     total_docs = []
   
#     for pdf_url in pdf_paths:
#         pdfreader =  PyPDFLoader(pdf_url)
#         pdf_doc = pdfreader.load_and_split()
#         doc = text_splitter.split_documents(pdf_doc)
#         print('각 pdf 별 chunk 수: ', len(doc))
#         for i in range(len(doc)):
#             print(doc[i].page_content)
#             extracted_content = create_extraction_chain(schema=raw_to_json_schema, llm=scraping_llm).invoke(doc[i].page_content)
#             print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
#             print(len(extracted_content))
#             print(extracted_content)
            
#             total_docs.append(extracted_content)
            
#         print('총 문서 수 업데이트: ', len(total_docs))
#     total_content = [str(item) for item in total_docs]
#     print('==========================================')
#     print('총 문서 수 업데이트: ', len(total_content))
 
#     docsearch = FAISS.from_texts(total_content, embeddings)
 
#     docsearch.embedding_function
#     docsearch.save_local(os.path.join(db_save_path, "cloud_bot_20240327_faiss_db"))


def process_document(pdf_content):
    """
    개별 PDF 문서를 처리하는 함수.
    pdf_content: 분할된 PDF 문서 내용.
    """
    doc = text_splitter.split_documents(pdf_content)
    total_contents = []
    for i in range(len(doc)):
        extracted_content = create_extraction_chain(schema=raw_to_json_schema, llm=scraping_llm).invoke(doc[i].page_content)
        total_contents.append(extracted_content)
    return total_contents

def offline_faiss_save(pdf_paths):
    total_docs = []

    with Pool(processes=int(os.cpu_count() / 2)) as pool:
        results = pool.map(process_document, [PyPDFLoader(pdf_url).load_and_split() for pdf_url in pdf_paths])

    for result in results:
        total_docs.append(result)
        print('문서 별 chunk 수: ', len(result))

    total_content = [str(item) for item in total_docs]
    print('==========================================')
    print('총 문서 수: ', len(total_content))

    docsearch = FAISS.from_texts(total_content, embeddings)
    docsearch.save_local(os.path.join(db_save_path, "cloud_bot_20240327_faiss_db"))
    
    
if __name__ == '__main__':
    offline_faiss_save(pdf_paths)