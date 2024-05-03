

import os
from dotenv import load_dotenv
import json
import http.client
from typing import List
from langchain_core.embeddings import Embeddings

from config import embedding_headers

load_dotenv()
HCX_TOKEN_HOST=os.getenv("HCX_TOKEN_HOST")
NCP_EMBEDDING_URL=os.getenv("NCP_EMBEDDING_URL")

class HCXEmbedding(Embeddings):
    def _send_request(self, text):
        headers = embedding_headers
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
