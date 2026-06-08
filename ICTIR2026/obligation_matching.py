import os
import sys
import csv
from typing import List
from pydantic import BaseModel, Field
from openai import OpenAI
from collections import defaultdict

# Increase the field size limit to the maximum possible integer size for your system
max_int = sys.maxsize
while True:
    try:
        csv.field_size_limit(max_int)
        break
    except OverflowError:
        # Decrease the limit until it fits within the C long integer size
        max_int = int(max_int / 10)

class MatchedIndex(BaseModel):
    idx1 : int = Field("Index from the first list")
    idx2 : int = Field("Index from the second list")

class ObligationMatching(BaseModel):
    identicalMeaning : List[MatchedIndex] = Field(description="A list of index pairs indicating which obligation from list 1 that is identical in meaning to an obligation from list 2.")
    oppositelMeaning : List[MatchedIndex] = Field(description="A list of index pairs indicating which  obligation from list 1 that is exactly opposite in meaning to an obligatiom from list list 2.")
    unmatchedList1 : List[int] = Field(description="A list of indexes from list 1 that have no matches in the previous two lists.")
    unmatchedList2 : List[int] = Field(description="A list of indexes from list 2 that have no matches in the previous two lists.")

def format_list(lst):
    result = ""
    for i in range(len(lst)):
        result += f"{i}) " + lst[i] + "\n"
    return result

def score(scoreStruct):
    if (2*len(scoreStruct['opposite'])+len(scoreStruct['unmatchedList1'])+len(scoreStruct['unmatchedList2'])+len(scoreStruct['identical'])) == 0:
        return 0
    return len(scoreStruct['identical']) / (2*len(scoreStruct['opposite'])+len(scoreStruct['unmatchedList1'])+len(scoreStruct['unmatchedList2'])+len(scoreStruct['identical']))

base_prompt=(
    "You are a helpful legal assistant that understands legal clauses."
    "You will be provided two lists of contractual obligations, rights, and responsibilities."
    "Your task is to align obligations that have exactly the same legal interpretation, those that have completely opposite (i.e., negation), and those that are unmatched from each list."
    "Ignore differences in entities between the two lists (e.g., assume the company names are the same). Treat all parties between the lists as if they were the same.")

def run_model(list1,list2,model):
    # First turn: Extract requests
    if 'gpt-5' in model:
        completion = client.beta.chat.completions.parse(
            model=model,
            response_format=ObligationMatching,
            messages=[
                {"role": "user", "content": f"{base_prompt}\n\nThe first list of obligations is:\n\n{list1}\n\nThe second list of obligations is:\n\n{list2}"}
            ],
        )
    else:
        completion = client.beta.chat.completions.parse(
            model=model,
            response_format=ObligationMatching,
            messages=[
                {"role": "user", "content": f"{base_prompt}\n\nThe first list of obligations is:\n\n{list1}\n\nThe second list of obligations is:\n\n{list2}"}
            ],
            temperature=0
        )
    event = completion.choices[0].message.parsed
    results = dict()
    results['unmatchedList1'] = event.unmatchedList1
    results['unmatchedList2'] = event.unmatchedList2
    results['identical'] = [ [pair.idx1,pair.idx2] for pair in event.identicalMeaning]
    results['opposite'] = [ [pair.idx1,pair.idx2] for pair in event.oppositelMeaning]
    usage = [completion.usage.prompt_tokens,completion.usage.completion_tokens]
    return results, usage

def rank_clauses(clauses):
    scores = defaultdict(list)
    for clause in clauses:
        scores[clause[2]].append(clause)
    ranked = sorted(scores.items(), key =lambda tup: tup[0],reverse=True)
    results = []
    for i in range(len(ranked)):
        clause_score = ranked[i]
        for clause in clause_score[1]:
            clause.append(i+1)
            results.append(clause)
    return results

if __name__ == "__main__":

    # Set up OpenAI API client
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"),timeout=1200)

    
    
    cnt=0
    headers = []
    headers=(["Doc","Obls","Score", "Rank"])
    data = []
    gold = ""
    model= sys.argv[3]
    fn = os.path.basename(sys.argv[1])
    out_file = open(sys.argv[2]+"/matched-"+model+"-"+fn,'w')
    writer = csv.writer(out_file)
    usage_writer = csv.writer(open(sys.argv[2]+"/usage-matched-"+model+"-"+fn,'w'))
    writer.writerow(headers)
    with open(sys.argv[1]) as query_file:
        reader = csv.reader(query_file)
        for idx,row in enumerate(reader):
            if idx == 0:
                print(row[0])
                gold = row[1]
            data.append([row[0],row[1]])
        simresults = []
        usages = []
        for n,elt in data:
            results,usage = run_model(gold,elt,model)
            sim = score(results)
            usages.append(usage)
            simresults.append([n,results,sim])
        ranks = [simresults[0]]
        simresults = rank_clauses(simresults[1:])
        ranks.extend(simresults)
        for usage in usages:
            usage_writer.writerow(usage)
        for result in ranks:
            writer.writerow(result)
        out_file.close()