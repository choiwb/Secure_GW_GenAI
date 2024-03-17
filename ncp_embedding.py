# -*- coding: utf-8 -*-

import os
import uuid
from dotenv import load_dotenv
import json
import http.client


load_dotenv()
HCX_API_KEY_PRIMARY_VAL=os.getenv("HCX_API_KEY_PRIMARY_VAL")
REQUEST_ID=str(uuid.uuid4())
HCX_EMBEDDING_API_KEY=os.getenv("HCX_EMBEDDING_API_KEY")

class HCXEmbedding:
    # def __init__(self, host, api_key, api_key_primary_val, request_id):
    #     self._host = host
    #     self._api_key = API_KEY
    #     self._api_key_primary_val = api_key_primary_val
    #     self._request_id = request_id
    
    def __init__(self):
        self.documents = []
        self.document_embeddings = []
    
    def _send_request(self, completion_request):
        headers = {
            'Content-Type': 'application/json; charset=utf-8',
            'X-NCP-CLOVASTUDIO-API-KEY': HCX_EMBEDDING_API_KEY,
            'X-NCP-APIGW-API-KEY': HCX_API_KEY_PRIMARY_VAL,
            'X-NCP-CLOVASTUDIO-REQUEST-ID': REQUEST_ID
        }

        conn = http.client.HTTPSConnection(os.getenv("HCX_TOKEN_HOST"))
        conn.request('POST', 'EMBEDDING MODEL PATH !!!!!!!!!!!!!!!!!!!!!', json.dumps(completion_request), headers)
        response = conn.getresponse()
        result = json.loads(response.read().decode(encoding='utf-8'))
        conn.close()
        return result

    def embed_query(self, completion_request):
        res = self._send_request(completion_request)
        if res['status']['code'] == '20000':
            return res['result']['embedding']
        else:
            return 'Error'
        
    def embed_documents(self, documents):
        try:
            self.documents = documents
            self.document_embeddings = [self.embed_query({"text": doc}) for doc in documents]
            return self.document_embeddings
        except:
            self.document_embeddings = [self.embed_query({"text": documents})]
            return self.document_embeddings




if __name__ == '__main__':
    embeddings = HCXEmbedding()

    request_data = json.loads("""{
    "text" : "안녕"
}""", strict=False)

    # response_query = embeddings.embed_query(request_data)
    # response_text = embeddings.embed_documents(request_data)
    # print('***************************')
    # print(response_text)
    # print('***************************')
    # print(response_query)




