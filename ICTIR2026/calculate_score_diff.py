import pandas as pd
import sys
import csv
import numpy as np
import sklearn

def toscore(rankings):
    return [100 - r for r in rankings]


def compute_score_diff(other,skipFirst):
    if skipFirst:
        other= other_ranks.drop([0])
    return other['Score'].diff().dropna().abs().tolist()

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

    other_ranks_file_prefix = sys.argv[1]
    p = 0.9
    skipFirst = False
    if len(sys.argv) == 4:
        skipFirst = True
    with open(sys.argv[2],'w') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(['Clause','Value'])
        for  clause_type in clause_types:
            other_ranks = read_ranks(other_ranks_file_prefix,clause_type)
            diffs = compute_score_diff(other_ranks,skipFirst)
            for diff in diffs:
                writer.writerow([clause_type,diff])