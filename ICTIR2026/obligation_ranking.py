import os
import sys
import csv
from typing import List
from pydantic import BaseModel, Field
from openai import OpenAI

import pandas as pd

# Increase the field size limit to the maximum possible integer size for your system
max_int = sys.maxsize
while True:
    try:
        csv.field_size_limit(max_int)
        break
    except OverflowError:
        # Decrease the limit until it fits within the C long integer size
        max_int = int(max_int / 10)

class RankedClause(BaseModel):
    name : str = Field("Clause name.")
    rank : int = Field("Rank of the clause.")
    reasons: List[str] = Field("A list of reasons for why the clause was given this rank.")

class ClauseRanking(BaseModel):
    rankedClauses : List[RankedClause] = Field("The ranked list of clauses.")

def format_list(lst):
    result = ""
    for i in range(len(lst)):
        result += f"{i}) " + lst[i] + "\n"
    return result

def score(scoreStruct):
    if (2*len(scoreStruct['opposite'])+len(scoreStruct['unmatchedList1'])+len(scoreStruct['unmatchedList2'])+len(scoreStruct['identical'])) == 0:
        return 0
    return len(scoreStruct['identical']) / (2*len(scoreStruct['opposite'])+len(scoreStruct['unmatchedList1'])+len(scoreStruct['unmatchedList2'])+len(scoreStruct['identical']))

base_prompt="""
You are a legal clause comparison engine. Your job is to compare a provided template clause, Template, to a list of other clauses provided. 

You will rank the additional clauses by their legal similarity to the template clause by providing a rank (with 1 being the most similar).
Clauses that are equivalently similar to the template clause may be given the same rank, even if they are different from each other. 

In your comparison, you should factor in the parties involved, the obligations, any carve-outs, and any differences in qualification (e.g., "provided that", "from time to time", "except") for these obligations.
You should also include a list of reasons for your determination. Each reason should be concise. 
"""

def run_model(template,clauses,model):
    # First turn: Extract requests
    if 'gpt-5' in model:
        completion = client.beta.chat.completions.parse(
            model=model,
            response_format=ClauseRanking,
            messages=[
                {"role": "user", "content": f"{base_prompt}\n\nTemplate:\n\n{template}\n\ncClauses to Rank:\n\n{clauses}"}
            ],
        )
    else:
        completion = client.beta.chat.completions.parse(
            model=model,
            response_format=ClauseRanking,
            messages=[
                {"role": "user", "content": f"{base_prompt}\n\nTemplate:\n\n{template}\n\ncClauses to Rank:\n\n{clauses}"}
            ],
            temperature=0
        )
    event = completion.choices[0].message.parsed
    results = [ [elt.name, elt.rank, elt.reasons] for elt in event.rankedClauses]#[ [pair.idx1,pair.idx2] for pair in event.oppositelMeaning]
    usage = [completion.usage.prompt_tokens,completion.usage.completion_tokens]
    return results,usage

if __name__ == "__main__":

    # Set up OpenAI API client
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"),timeout=1200)

    
    
    cnt=0
    headers = []
    headers=(["Doc","Rank", "Reasons"])
    data = []
    gold = ""
    model= sys.argv[3]
    fn = os.path.basename(sys.argv[1])
    out_file = open(sys.argv[2]+"/ranked"+"-"+model+"-"+fn,'w')
    writer = csv.writer(out_file)
    usage_writer = csv.writer(open(sys.argv[2]+"/usage-ranked"+"-"+model+"-"+fn,'w'))
    writer.writerow(headers)
    with open(sys.argv[1]) as query_file:
        reader = csv.reader(query_file)
        for idx,row in enumerate(reader):
            if idx == 0:
                continue
            if idx == 1:
                print(row[0])
                gold = row[1]
                continue
            data.append([row[0],row[1]])
        clauses = "\n\n".join([f"Clause Name:{name}\nClause:\n{clause}" for name,clause in data])
        results,usage = run_model(gold,clauses,model)
        results = sorted(results, key=lambda tup: tup[1])
        usage_writer.writerow(usage)
        for result in results:
            writer.writerow(result)

        out_file.close()