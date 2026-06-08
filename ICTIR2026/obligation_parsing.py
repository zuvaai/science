import os
import sys
import json
import csv
from typing import List
from pydantic import BaseModel, Field, create_model
from openai import OpenAI

import pandas as pd

class ObligationList(BaseModel):
    obligations : List[str] = Field(description="A list of each party's obligations, rights, or responsibilities based upon the provided clause text.")


base_prompt = ("Your task is to analyze the clause as written, based solely on the operative content, not the heading."
                "List all rights, responsibilities, and obligations based only on what the full text actually states."
                "The clause may contain a heading, but this heading may not reflect the actual rights, responsibilities, or obligations in the operative language."
                "Each list item must be semantically exclusive and independent."
                "Do not infer, interpret, or correct errors. Use the clause exactly as written."
                "Do not apply legal norms or drafting conventions."
                "Several examples of clauses and expected obligations follows: "
                "Example 1: "
                "Clause: Exclusivity. During the Term. SIEMENS may develop, manufacture or commercialize a Galectin-3 assay other than Product even if such other assay would fall within the claims in the patents included in the Patent Rights and for so long as BGM holds all Patent Rights and intellectual property necessary to sell Products subject to these claims."
                "Obligations: [\"SIEMENS has the right to develop, manufacture, or commercialize a Galectin-3 assay other than the Product during the Term.\", \"SIEMENS may develop, manufacture, or commercialize such other Galectin-3 assay even if it would fall within the claims in the patents included in the Patent Rights.\", \"SIEMENS's right to develop, manufacture, or commercialize such other Galectin-3 assay is contingent upon BGM holding all Patent Rights and intellectual property necessary to sell Products subject to these claims.\"]"
                "Example 2: "
                "Clause: Assignment.  This Agreement may be assigned or otherwise transferred by either Party without the prior written consent of the other Party; provided, however, that (a) LipoScience may not, without such consent from Agilent, assign its rights together with its obligations under this Agreement to any entity acquiring substantially all of its assets or stock, or its business relating to the Vantera Analyzer, or with which it may merge and (b) Agilent may not, without consent from LipoScience, assign its rights together with its obligations under this Agreement to any entity acquiring substantially all the assets of its NMR business, or with which it may merge; provided, however, that such Party’s rights and obligations under this Agreement shall not be assumed by its successor in interest in any such transaction. Any purported assignment in violation of the preceding sentence shall be allowed. Any permitted assignee shall not assume all obligations of its assignor under this Agreement."
                "Obligations: [\"Either Party may assign or transfer the Agreement without the prior written consent of the other Party.\", \"LipoScience may not assign its rights and obligations under the Agreement to any entity acquiring substantially all of its assets or stock, or its business relating to the Vantera Analyzer, or with which it may merge, without Agilent's consent.\", \"Agilent may not assign its rights and obligations under the Agreement to any entity acquiring substantially all the assets of its NMR business, or with which it may merge, without LipoScience's consent.\", \"A Party's rights and obligations under the Agreement shall not be assumed by its successor in interest in any transaction involving assignment or transfer.\", \"Any purported assignment in violation of the clause is allowed.\", \"Any permitted assignee shall not assume all obligations of its assignor under the Agreement.\"]"
                "Example 3:"
                "Clause: RECRUITMENT. During the term of employment and for a period of 12 months following termination of employment for any reason other than a Change of Control Termination, Executive may directly or indirectly hire any of Ceridian's employees who are employed by businesses for which Executive has or had management responsibility, or solicit any of Ceridian's employees who are employed by businesses for which Executive has or had management responsibility for the purpose of hiring them or inducing them to leave their employment with Ceridian, or may Executive own, manage, operate, join, control, consult with, participate in the ownership, management, operation or control of, be employed by, or be connected in any manner with any person or entity which engages in the conduct proscribed in this Section 6.03. This provision shall not preclude Executive from responding to a request (other than by Executive's employer) for a reference with respect to an individual's employment qualifications."
                "Obligations: [\"The Executive may directly or indirectly hire any of Ceridian's employees who are employed by businesses for which the Executive has or had management responsibility during the term of employment and for a period of 12 months following termination of employment for any reason other than a Change of Control Termination.\", \"The Executive may solicit any of Ceridian's employees who are employed by businesses for which the Executive has or had management responsibility for the purpose of hiring them or inducing them to leave their employment with Ceridian during the term of employment and for a period of 12 months following termination of employment for any reason other than a Change of Control Termination.\", \"The Executive may own, manage, operate, join, control, consult with, participate in the ownership, management, operation or control of, be employed by, or be connected in any manner with any person or entity which engages in the conduct proscribed in this Section 6.03 during the term of employment and for a period of 12 months following termination of employment for any reason other than a Change of Control Termination.\", \"The provision does not preclude the Executive from responding to a request (other than by the Executive's employer) for a reference with respect to an individual's employment qualifications.\"]\n")

def run_model(model,content):
    # First turn: Extract requests
    if "gpt-5" in model:
        completion = client.beta.chat.completions.parse(
            model=model,
            response_format=ObligationList,
            messages=[
                {"role": "user", "content": f"{base_prompt}\n\nClause to analyze:\n\n{content}"},
            ],
        )
    else:
        completion = client.beta.chat.completions.parse(
            model=model,
            response_format=ObligationList,
            messages=[
                {"role": "user", "content": f"{base_prompt}\n\nClause to analyze:\n\n{content}"},
            ],
            temperature=0,
        )
    event = completion.choices[0].message.parsed
    usage = [completion.usage.prompt_tokens,completion.usage.completion_tokens]
    return json.dumps(event.obligations),usage

if __name__ == "__main__":

    # Set up OpenAI API client
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    model=sys.argv[3]

    fn = os.path.basename(sys.argv[1])
    out_file = open(sys.argv[2]+"/"+model+"-"+fn,'w')
    writer = csv.writer(out_file)
    usage_writer = csv.writer(open(sys.argv[2]+"/usage"+model+"-"+fn,'w'))
    cnt=0
    with open(sys.argv[1]) as query_file:
        reader = csv.reader(query_file)
        for row in reader:
            if cnt == 0:
                results = [row[0],row[1]]
                print(results)
                cnt += 1
                continue
            name = row[0]
            results = [name]
            result,usage = run_model(model, row[1])
            usage_writer.writerow(usage)
            results.append(result)
            writer.writerow(results)
            cnt+=1
    out_file.close()