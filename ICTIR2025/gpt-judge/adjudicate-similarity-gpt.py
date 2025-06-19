from openai import OpenAI
import csv
import sys
import os
from tenacity import retry,wait_exponential

prompt = """You are to determine whether the following legal clauses have the same meaning or whether they are different. Respond only with the word "Same" or "Different". If you cannot make a determination respond with only the word "Unsure".

Clause 1:
{0}

Clause 2:
{1}"""


client = OpenAI(
    # Run as OPENAI_API_KEY=<YOUR KEY> python3 adjudicate-similarity_gpt.py
    api_key=os.environ.get("OPENAI_API_KEY"),
)

@retry(wait=wait_exponential(multiplier=1,min=1,max=30))
def run_model(model,c1,c2):
    response = client.chat.completions.create(model=model, messages=[
        {
            'role': 'user',
            'content': prompt.format(c1,c2) ,
        },
    ])
    return response.choices[0].message.content.replace('\0','')
    

models = ["gpt-4o-mini","o1-mini","o3-mini"]
out_file = open(sys.argv[2],'w')
writer = csv.writer(out_file)
cnt=0
with open(sys.argv[1]) as query_file:
    reader = csv.reader(query_file)
    for row in reader:
        if cnt == 0:
            writer.writerow(row)
            cnt += 1
            continue
        for model in models:
            cols = [model]
            results = run_model(model,row[0],row[1])
            cols.append(results)
            results = run_model(model,row[0],row[2])
            cols.append(results)
            writer.writerow(cols)
            cnt+=1
out_file.close()