

import os
import uuid
from dotenv import load_dotenv
import json
import http.client
from typing import List
from langchain_core.embeddings import Embeddings


load_dotenv()
HCX_API_KEY_PRIMARY_VAL=os.getenv("HCX_API_KEY_PRIMARY_VAL")
REQUEST_ID=str(uuid.uuid4())
HCX_EMBEDDING_API_KEY=os.getenv("HCX_EMBEDDING_API_KEY")
HCX_TOKEN_HOST=os.getenv("HCX_TOKEN_HOST")
NCP_EMBEDDING_URL=os.getenv("NCP_EMBEDDING_URL")

class HCXEmbedding(Embeddings):
    def _send_request(self, text):
        headers = {
            'Content-Type': 'application/json; charset=utf-8',
            'X-NCP-CLOVASTUDIO-API-KEY': HCX_EMBEDDING_API_KEY,
            'X-NCP-APIGW-API-KEY': HCX_API_KEY_PRIMARY_VAL,
            'X-NCP-CLOVASTUDIO-REQUEST-ID': REQUEST_ID
        }
        completion_request = {'text': text}
        conn = http.client.HTTPSConnection(HCX_TOKEN_HOST)
        conn.request('POST', NCP_EMBEDDING_URL, json.dumps(completion_request), headers)
        response = conn.getresponse()
        result = json.loads(response.read().decode('utf-8'))
        conn.close()
        return result
        
    def embed_documents(self, documents: List[str]) -> List[List[float]]:
        embeddings = []
        for doc_text in documents:
            res = self._send_request(doc_text)
            if res['status']['code'] == '20000':
                embedding = res['result']['embedding']
                embeddings.append(embedding)
        return embeddings
    
    def embed_query(self, query: str) -> List[float]:
        return self.embed_documents([query])[0]
