import pandas as pd
import sys
import csv
import numpy as np
import sklearn

def toscore(rankings):
    return [100 - r for r in rankings]


def compute_ami(human_ranks,other_ranks,skipFirst):
    gold = human_ranks.sort_values(by=['Doc'])
    gold = gold['Rank'].tolist()
    if skipFirst:
        other_ranks = other_ranks.drop([0])
    pred = other_ranks.sort_values(by=['Doc'])
    pred = pred['Rank'].tolist()
    return sklearn.metrics.adjusted_mutual_info_score(gold,pred)


def read_ranks(prefix,clause_type):
    data = pd.read_csv(f"{prefix}{clause_type}.csv")
    return data

if __name__ == "__main__":
    clause_types = [
        "assignment",
        "confidentiality",
        "force-majeure",
        "indemnity",
        "publicity",
        "duration",
        "non-disparagement",
        "non-compete",
        "notice",
        "termination",
    ]

    human_ranks_file_prefix = sys.argv[1]
    other_ranks_file_prefix = sys.argv[2]
    p = 0.9
    skipFirst = False
    if len(sys.argv) == 5:
        skipFirst = True
    with open(sys.argv[3],'w') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(['Clause','Value'])
        for  clause_type in clause_types:
            human_ranks = read_ranks(human_ranks_file_prefix,clause_type)
            other_ranks = read_ranks(other_ranks_file_prefix,clause_type)
            row = compute_ami(human_ranks, other_ranks,skipFirst)
            writer.writerow([clause_type,row])
    