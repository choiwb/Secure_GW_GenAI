# -*- coding: utf-8 -*-

import uuid
import base64
import json
import http.client




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
            'X-NCP-CLOVASTUDIO-REQUEST-ID': self._request_id
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



token_completion_executor = token_CompletionExecutor(
      host='HOST !!!!!!!!!!!!!!!!!!!!!!!!1',
      api_key='API KEY !!!!!!!!!!!!!!!!!!!!!!!!1',
      api_key_primary_val = 'API KEY PRIMARY VAL !!!!!!!!!!!!!!!!!!!!!!!!1',
      request_id=str(uuid.uuid4())
)
