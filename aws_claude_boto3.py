
from LCEL import aws_retrieval_qa_chain

prompt = 'window injection과 관련된 악성코드는 뭐가 있어?'

for chunk in aws_retrieval_qa_chain.stream({"question": prompt}):
    print(chunk, end = "", flush = True)
