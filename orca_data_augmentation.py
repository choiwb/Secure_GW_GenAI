

import os
import time
import traceback
from dotenv import load_dotenv
from multiprocessing import Pool
import pandas as pd
import openai


# .env 파일 로드
load_dotenv()

os.getenv('OPENAI_API_KEY')


df = pd.read_excel('YOUR DATASET !!!!!!!!!!!!!!!!!')

orca_prompt_list = ['You are an cyber security analyst. Provide a detailed answer so user don’t need to search outside to understand the answer.',
'You are an cyber security analyst. You will be given a task. You must generate a detailed and long answer.',
'You are an cyber security analyst, who always provide explanation. Think like you are answering to a five year old.',
'You are an cyber security analyst that follows instruction extremely well. Help as much as you can.',
'You are an cyber security analyst that helps people find information. Provide a detailed answer so user don’t need to search outside to understand the answer.',
'You are an cyber security analyst. User will you give you a task. Your goal is to complete the task as faithfully as you can. While performing the task think step-by-step and justify your steps.', 
'You should describe the task and explain your answer. While answering a multiple choice question, first output the correct answer(s). Then explain why other answers are wrong. Think like you are answering to a five year old.',
'Explain how you used the definition to come up with the answer.',
'You are an cyber security analyst. You should describe the task and explain your answer. While answering a multiple choice question, first output the correct answer(s). Then explain why other answers are wrong. You might need to use additional knowledge to answer the question.',
'You are an cyber security analyst that helps people find information. User will you give you a question. Your task is to answer as faithfully as you can. While answering think step-bystep and justify your answer.',
'User will you give you a task with some instruction. Your job is follow the instructions as faithfully as you can. While answering think step-by-step and justify your answer.',
'You are a cyber security analyst. Given a task, you explain in simple steps what the task is asking, any guidelines it provides and how to use those guidelines to find the answer.',
'You are an cyber security analyst, who knows every language and how to translate one language to another. Given a task, you explain in simple steps what the task is asking, any guidelines that it provides. You solve the task and show how you used the guidelines to solve the task.',
'Given a definition of a task and a sample input, break the definition into small parts. Each of those parts will have some instruction. Explain their meaning by showing an example that meets the criteria in the instruction. Use the following format: Part #: a key part of the definition. Usage: Sample response that meets the criteria from the key part. Explain why you think it meets the criteria.',
'You are an cyber security analyst that helps people find information.']

orca_prompt_df = pd.DataFrame(orca_prompt_list, columns=['orca_system_prompt'])
orca_prompt_aug_df = pd.concat([orca_prompt_df] * len(df), ignore_index=True)

# df를 orca_prompt_list 만큼 곱하기
# df의 각 행 별로 5번씩 복붙하기
df = pd.concat([df] * len(orca_prompt_list), ignore_index=True)
df = df.sort_values(by = 'input', ascending = True, ignore_index=True)

df = pd.concat([df, orca_prompt_aug_df], axis = 1)
df['orca_생성_답변'] = ''


aug_df = pd.DataFrame()

# Calculate the number of DataFrames to create
each_sampling_df_len = 100
num_dataframes = (len(df) // each_sampling_df_len) + 1 if len(df) % each_sampling_df_len != 0 else len(df) // each_sampling_df_len

# Split the DataFrame into smaller DataFrames and create individual variables
for i in range(num_dataframes):
    start_idx = i * each_sampling_df_len
    end_idx = min((i + 1) * each_sampling_df_len, len(df))
    globals()[f'df{i+1}'] = df[start_idx:end_idx].copy()
    print(globals()[f'df{i+1}'].shape)


def chatgpt_orca_answer(row):
    index, data = row
    context = '공격 명은' + ' ' + data['input'] + ' ' + '일 때, 공격 명에 대한 설명은' + ' ' + data['output']
    ques = '입력된 공격 명에 대한 설명을 한국어로 작성해주세요.'
    max_tokens = 1024
    # Calculate the remaining tokens available for the answer
    remaining_tokens = max_tokens - len(openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": data['orca_system_prompt']},
            {"role": "assistant", "content": context},
            {"role": "user", "content": ques}
        ]
    )['choices'][0]['message']['content'].split())
    
    completion = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    max_tokens=remaining_tokens,
    messages=[
        {"role": "system", "content": data['orca_system_prompt']},
        {"role": "assistant", "content": context},
        {"role": "user", "content": ques}
    ]
    )
    return completion



def process_row(row):
    index, data = row

    try:
        completion = chatgpt_orca_answer(row)

        answer = completion['choices'][0]['message']['content']
        answer = answer.lower().replace('\n', ' ')
        return answer
    except openai.error.RateLimitError as e:
        print("Rate limit error: ", e)
        print("Waiting for 1 minute before retrying...")
        pass


def process_row_with_retry(row, max_retries=3, sleep_interval=60):
    for i in range(max_retries):
        try:
            return process_row(row)
        except openai.error.APIError as e:
            print(f"Error occurred: {e}. Retrying ({i+1}/{max_retries}) after {sleep_interval} seconds...")
            time.sleep(sleep_interval)  # prevent too many requests in a short time
        except Exception as e:  # catch other exceptions
            print(f"Unexpected error occurred: {traceback.format_exc()}")
    return None  # or you may want to return a special value indicating error

start = time.time()

for i in range(num_dataframes):

    with Pool(int(os.cpu_count() / 4)) as pool:
        answers = pool.map(process_row_with_retry, globals()[f'df{i+1}'].iterrows())

    globals()[f'df{i+1}']['orca_생성_답변'] = answers    
    print('%d번 째 100개 데이터셋 답변 완료' %(i+1))
    # aug_df 에 globals()[f'df{i+1}'] 를 concat
    aug_df = pd.concat([aug_df, globals()[f'df{i+1}']], axis=0)
    print(aug_df.shape)

end = time.time()
print('답변 생성 소요 시간: %.2f (초)' %(end - start))



aug_df.to_excel('SAVING YOUR DATASET !!!!!!!!!!!!!!!!!', index=False)
