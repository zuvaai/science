import ollama
import csv
import sys

prompt = """You are to determine whether the following legal clauses have the same meaning or whether they are different. Respond only with the word "Same" or "Different". If you cannot make a determination respond with only the word "Unsure".

Clause 1:
{0}

Clause 2:
{1}"""


def run_model(model,c1,c2):
    response = ollama.chat(model=model, messages=[
        {
            'role': 'user',
            'content': prompt.format(c1,c2),
        },
    ])
    raw = response['message']['content'].replace('\0','')
    idx = raw.find("</think>")
    ans = raw[idx+8:]
    return [raw,ans]

models = ["deepseek-r1:1.5b","deepseek-r1:8b","deepseek-r1:32b"]
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
            cols.extend(results)
            results = run_model(model,row[0],row[2])
            cols.extend(results)
            writer.writerow(cols)
            cnt+=1
out_file.close()