from openai import OpenAI
import csv
import sys
import os
from tenacity import retry,wait_exponential

prompt = """\"Rewrite the following contract clause so that the syntax of both its title (if any) and body are significantly altered while preserving its legal meaning. The rewritten clause should use different sentence structures, alternative phrasing, and varied word choices, but it must not change the intent, obligations, or rights described. Avoid simply replacing words with synonyms—focus on restructuring sentences while maintaining clarity and legal precision. If the original clause includes a title, provide a reworded version of the title. However, if no title is present, do not create one.\"
Clause:
{0}"""


client = OpenAI(
    # Run as OPENAI_API_KEY=<YOUR KEY> python3 gen-variants.py
    api_key=os.environ.get("OPENAI_API_KEY"),
)

@retry(wait=wait_exponential(multiplier=1,min=1,max=30))
def run_model(model,clause,iterations):
    responses = []
    for i in range(iterations):
        response = client.chat.completions.create(model=model, messages=[
            {
                'role': 'user',
                'content': prompt.format(clause) ,
            },
        ])
        responses.append(response.choices[0].message.content.replace('\0',''))
    return responses

contexts={}

models = ["gpt-4o"]
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
                cols = row[0:1]
                cols.extend([model])
                results = run_model(model,row[0],1)
                cols.extend(results)
                writer.writerow(cols)
        cnt += 1
out_file.close()