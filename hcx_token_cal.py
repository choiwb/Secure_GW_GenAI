
import os
import json
import http.client
from dotenv import load_dotenv

from config import token_headers, sec_headers

# .env 파일 로드
load_dotenv()
token_host = os.getenv('HCX_TOKEN_HOST')
token_url = os.getenv('HCX_TOKEN_URL')

class token_CompletionExecutor:
    def _send_request(self, completion_request):
        token_sec_headers = token_headers | sec_headers
        
        conn = http.client.HTTPSConnection(token_host)
        conn.request('POST', token_url, json.dumps(completion_request), token_sec_headers)

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


