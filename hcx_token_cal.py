# -*- coding: utf-8 -*-

import os
import uuid
import json
import http.client
from dotenv import load_dotenv


class token_CompletionExecutor:
    def __init__(self, host, api_key, api_key_primary_val, request_id):
        self._host = host
        self._api_key = api_key
        self._api_key_primary_val = api_key_primary_val
        self._request_id = request_id

    def _send_request(self, completion_request):
        headers = {
            'Content-Type': 'application/json; charset=utf-8',
            'X-NCP-CLOVASTUDIO-API-KEY': self._api_key,
            'X-NCP-APIGW-API-KEY': self._api_key_primary_val,
            'X-NCP-CLOVASTUDIO-REQUEST-ID': self._request_id,
            # 응답에 보안 헤더 추가
            'Strict-Transport-Security': 'max-age=63072000; includeSubdomains; preload',
            'X-Content-Type-Options': 'nosniff',
            'Content-Security-Policy': "default-src 'none'; img-src 'self'; script-src 'self'; style-src 'self'; object-src 'none'; frame-ancestors 'none'",
            'referrer-policy': 'same-origin'
        }

        conn = http.client.HTTPSConnection(self._host)
        conn.request('POST', 'MODEL PATH !!!!!!!!!!!!!', json.dumps(completion_request), headers)

        response = conn.getresponse()
        result = json.loads(response.read().decode(encoding='utf-8'))
        conn.close()
        return result

    def execute(self, completion_request):
        res = self._send_request(completion_request)
        if res['status']['code'] == '20000':
            return res['result']['messages']
        else:
            return 'Error'



# .env 파일 로드
load_dotenv()

token_completion_executor = token_CompletionExecutor(
      host=os.getenv('HCX_TOKEN_HOST'),
      api_key=os.getenv('HCX_TOKEN_API_KEY'),
      api_key_primary_val=os.getenv('HCX_API_KEY_PRIMARY_VAL'),
      request_id=str(uuid.uuid4())
)
