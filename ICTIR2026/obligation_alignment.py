import os
import sys
import csv
from typing import List
from pydantic import BaseModel, Field
from openai import OpenAI
from collections import defaultdict

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

class MatchedIndex(BaseModel):
    idx1 : int = Field("Index from the first list")
    idx2 : int = Field("Index from the second list")

class AlignedObligation(BaseModel):
    atom : str = Field(description="The identified obligation atom")
    clause_a : List[int] = Field(description="The sentence identifiers of all sentences that correspond to the identified atom from clause_a.")
    clause_b : List[int] = Field(description="The sentence identifiers of all sentences that correspond to the identified atom from clause_b.")

class UnAlignedObligation(BaseModel):
    atom : str = Field(description="The obligation atom identified from the clause.")
    sentence_ids : List[int]  = Field(description="The sentence ids from the clause that contain the obligation atom.")
class ObligationMatching(BaseModel):
    identicalMeaning : List[AlignedObligation]= Field(description="A list of aligned obligation atoms from clause_a and clause_b that have equivalent legal meaning.")
    oppositeMeaning : List[AlignedObligation] = Field(description="A list of aligned obligation atoms from clause_a and clause_b that have opposite legal meaning.")
    unmatchedClauseA : List[UnAlignedObligation] = Field(description="A list of obligation atoms unique to clause_a.")
    unmatchedClauseB : List[UnAlignedObligation] = Field(description="A list of obligation atoms unique to clause_a.")

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
You are a legal clause comparison engine. Your job is to compare Clause A and Clause B by extracting and aligning their obligations and returning them.

DEFINITIONS
- “Obligation” means any legally operative requirement, prohibition, permission, or commitment. This includes: shall/must/will/agree to, may/is permitted to, shall not/may not/is prohibited from, is entitled to, is responsible for, must ensure, will cause.
- “Legally equivalent obligations” means that, after normalizing defined terms and synonyms, the parties (actor/beneficiary), the deontic modality (must/may/must not), the action, and all material conditions/exceptions are the same in legal effect.
- “Opposite meaning” means the obligations are in direct conflict in legal effect (e.g., permission vs prohibition for same action; required vs prohibited; required vs expressly not required; “may disclose” vs “shall not disclose”), or they reverse a key condition/exception such that the effect flips.
- “Unmatched obligation” means an obligation extracted from one clause has no corresponding aligned obligation in the other clause (neither same nor opposite).
- “Sentence evidence”:
  - For SAME or OPPOSITE: include the sentence(s) from Clause A and the sentence(s) from Clause B where the aligned obligations appear.
  - For UNMATCHED: include the sentence(s) from the clause where the obligation appears.

INPUTS
You will be given:
- clause_a: a string
- clause_b: a string

INSTRUCTIONS
1) Sentence split each clause. Use the original sentences verbatim for evidence (no rewriting). If a single obligation spans multiple sentences, include all relevant sentences.
2) Extract “obligation atoms” from each clause:
   - Normalize each atom into a structured representation:
     actor (who must/may/must not), action (what), modality (MUST / MAY / MUST_NOT), object/recipient (to whom/what), conditions (if/when/unless/except), time limits, geographic limits, purpose limitations, confidentiality/publicity constraints, approvals/consents, and remedies (if stated as duties).
   - Create one atom per distinct duty/permission/prohibition. If a sentence contains multiple duties, split them.
3) Align obligations across Clause A and Clause B:
   - Produce pairs when they refer to the same underlying action/subject matter and same actor/beneficiary context.
   - Classify each aligned pair as having the same meaning or opposite meaning based on legal effect, not wording.
   - If partially overlapping, do NOT force SAME. Prefer:
       - SAME only if all material elements match.
       - OPPOSITE if the net legal effect conflicts.
       - Partial overlaps should be split into discrete obligation atoms to maximize alignment. 
       - Carve-outs should be treated as separate obligation atom. 
4) Unmatched obligations:
   - List all atoms from Clause A with no alignment in Clause B, and all atoms from Clause B with no alignment in Clause A.
5) For every result item, include:
   - A concise “obligation atom” in neutral legal language (short, not a quotation).
   - The sentence id of the corresponding clause sentence(s). 
6) Quality rules:
   - Do not hallucinate parties or conditions not present.
   - If parties are implicit (e.g., “each party”), use that phrasing.
   - Treat defined terms as-is; do not expand beyond given text.

MATERIAL QUALIFIERS RULE (STRICT):
Treat any explicit qualifier as material (e.g., "from time to time", "provided that", "except", "during the term", "solely"). Do NOT normalize them away.
If Clause A and Clause B express the same core obligation but one side adds/removes a material qualifier, then:
    - Do NOT put the pair in identicalMeaning.
    - Align only the core shared obligation in identicalMeaning (if the core text is truly the same),
    -Create an unmatched obligation atom for the qualifier on the side where it appears, using the exact sentence ids.
"""

def run_model(list1,list2,model):
    # First turn: Extract requests
    if 'gpt-5' in model:
        completion = client.beta.chat.completions.parse(
            model=model,
            response_format=ObligationMatching,
            messages=[
                {"role": "user", "content": f"{base_prompt}\n\nclause_a:\n\n{list1}\n\nclause_b:\n\n{list2}"}
            ],
        )
    else:
        completion = client.beta.chat.completions.parse(
            model=model,
            response_format=ObligationMatching,
            messages=[
                {"role": "user", "content": f"{base_prompt}\n\nclause_a:\n\n{list1}\n\nclause_b:\n\n{list2}"}
            ],
            temperature=0
        )
    event = completion.choices[0].message.parsed
    results = dict()
    results['unmatchedList1'] = [ [elt.atom, elt.sentence_ids] for elt in event.unmatchedClauseA]
    results['unmatchedList2'] = [ [elt.atom, elt.sentence_ids] for elt in event.unmatchedClauseB]
    results['identical'] = [ [elt.atom,elt.clause_a, elt.clause_b] for elt in event.identicalMeaning]
    results['opposite'] = [ [elt.atom,elt.clause_a, elt.clause_b] for elt in event.oppositeMeaning]
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
    headers=(["Doc","Obls","Score","Rank"])
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
                continue
            if idx == 1:
                gold = row[1]
            data.append([row[0],row[1]])
        simresults = []
        usages = []
        for n,elt in data:
            results, usage = run_model(gold,elt,model)
            usages.append(usage)
            sim = score(results)
            simresults.append([n,results,sim])
        ranks = [simresults[0]]
        simresults = rank_clauses(simresults[1:])
        ranks.extend(simresults)
        for usage in usages:
            usage_writer.writerow(usage)
        for result in ranks:
            writer.writerow(result)
        out_file.close()