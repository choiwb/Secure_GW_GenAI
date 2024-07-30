
import os
import time
import asyncio

from vector_db import offline_chroma_save, offline_pgvector_save, offline_milvus_save, async_offline_milvus_save
from config import pdf_folder_path, db_name


pdf_paths = []
for filename in os.listdir(pdf_folder_path):
    if filename.endswith(".pdf"):
        # 완전한 파일 경로 생성
        pdf_path = os.path.join(pdf_folder_path, filename)
        pdf_paths.append(pdf_path)

start = time.time()
# total_content = offline_chroma_save(pdf_paths, db_name)
# total_content = offline_pgvector_save(pdf_paths)
# total_content = offline_milvus_save(pdf_paths)
total_content = asyncio.run(async_offline_milvus_save(pdf_paths))

end = time.time()
print('임베딩 완료 시간: %.2f (초)' %(end-start))
