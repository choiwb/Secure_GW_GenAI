# -*- coding: utf-8 -*-

import os
import uuid
from dotenv import load_dotenv
import json
import http.client
from chromadb import Documents, EmbeddingFunction, Embeddings

load_dotenv()
HCX_API_KEY_PRIMARY_VAL=os.getenv("HCX_API_KEY_PRIMARY_VAL")
REQUEST_ID=str(uuid.uuid4())
HCX_EMBEDDING_API_KEY=os.getenv("HCX_EMBEDDING_API_KEY")
HCX_TOKEN_HOST=os.getenv("HCX_TOKEN_HOST")


class HCXEmbedding(EmbeddingFunction):

    def _send_request(self, text):
        headers = {
            'Content-Type': 'application/json; charset=utf-8',
            'X-NCP-CLOVASTUDIO-API-KEY': HCX_EMBEDDING_API_KEY,
            'X-NCP-APIGW-API-KEY': HCX_API_KEY_PRIMARY_VAL,
            'X-NCP-CLOVASTUDIO-REQUEST-ID': REQUEST_ID
        }
        completion_request = {'text': text}  # 예제 요청 포맷, 실제 요청 포맷에 맞춰 수정 필요
        conn = http.client.HTTPSConnection(HCX_TOKEN_HOST)
        conn.request('POST', '/testapp/v1/api-tools/embedding/clir-emb-dolphin/266736b2551c43ff83fd1d18a524923b', json.dumps(completion_request), headers)
        response = conn.getresponse()
        result = json.loads(response.read().decode('utf-8'))
        conn.close()
        return result

    def __call__(self, input: Documents) -> Embeddings:
        embeddings = []
        for text in input.documents:
            try:
                res = self._send_request(text)
                if res['status']['code'] == '20000':
                    embedding = res['result']['embedding']
                    embeddings.append(embedding)
                else:
                    print('Error retrieving embedding')
                    return None
            except Exception as e:
                print(f"Error in E5EmbeddingFunction: {e}")
                return None

        return Embeddings(embeddings)
    
    # embed_query 메서드 추가
    def embed_query(self, query):
        # 단일 쿼리에 대한 임베딩 생성
        res = self._send_request(query)
        if res['status']['code'] == '20000':
            return res['result']['embedding']
        else:
            print('Error retrieving embedding for query')
            return None
        
    def embed_documents(self, documents):
        """여러 문서에 대해 임베딩을 생성하고 반환합니다.

        Args:
            documents (list of str): 임베딩을 생성할 문서들의 리스트.

        Returns:
            list of embeddings: 생성된 임베딩들의 리스트. 각 임베딩은 문서에 대응됩니다.
        """
        embeddings = []
        for doc_text in documents:
            try:
                res = self._send_request(doc_text)
                if res['status']['code'] == '20000':
                    embedding = res['result']['embedding']
                    embeddings.append(embedding)
                else:
                    print('Error retrieving embedding for document')
                    embeddings.append(None)  # 임베딩 생성 실패시 None 추가
            except Exception as e:
                print(f"Error in embed_documents: {e}")
                embeddings.append(None)
        return embeddings





