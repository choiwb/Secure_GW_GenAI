
import os
import time

from vector_db import offline_chroma_save, ncp_offline_chroma_save, offline_pgvector_save
from config import pdf_folder_path


pdf_paths = []
for filename in os.listdir(pdf_folder_path):
    if filename.endswith(".pdf"):
        # 완전한 파일 경로 생성
        globals()[f'pdf_path_{filename}']  = os.path.join(pdf_folder_path, filename)
        # print(globals()[f'pdf_path_{filename}'])
        pdf_paths.append(globals()[f'pdf_path_{filename}'])

start = time.time()
# total_content = offline_chroma_save(pdf_paths)
total_content = ncp_offline_chroma_save(pdf_paths)
# total_content = offline_pgvector_save(pdf_paths)
end = time.time()
print('임베딩 완료 시간: %.2f (초)' %(end-start))
