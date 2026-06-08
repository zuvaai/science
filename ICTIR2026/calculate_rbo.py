import pandas as pd
import sys
import csv


from collections import defaultdict
from math import isclose
import pandas as pd

def tie_groups(values, items, ascending=True):
    """
    Turn a scored list into ordered tie-buckets.
    Returns a list of (score, [items...]) sorted by score.
    """
    pairs = list(zip(values, items))
    pairs.sort(key=lambda x: x[0], reverse=not ascending)

    groups = []
    for score, it in pairs:
        if not groups or score != groups[-1][0]:
            groups.append((score, [it]))
        else:
            groups[-1][1].append(it)
    return groups

def expected_topd_indicator(tie_groups, d):
    """
    For a ranking expressed as ordered tie-buckets, return inclusion probabilities
    for the top-d positions under uniform random tie-breaking within buckets.
    """
    probs = {}
    seen = 0
    for _, bucket in tie_groups:
        b = len(bucket)
        if seen >= d:
            break

        if seen + b <= d:
            # entire bucket is surely included
            for it in bucket:
                probs[it] = 1.0
            seen += b
        else:
            # boundary bucket: only (d-seen) of b items make it in, uniformly
            take = d - seen
            p = take / b
            for it in bucket:
                probs[it] = p
            seen = d
            break

    return probs

def expected_overlap_at_depth(tie_groups_A, tie_groups_B, d):
    """
    Expected |Top_d(A) ∩ Top_d(B)| under independent uniform tie-breaking within ties.
    """
    pA = expected_topd_indicator(tie_groups_A, d)
    pB = expected_topd_indicator(tie_groups_B, d)

    # Items missing from dict have prob 0
    items = set(pA.keys()) | set(pB.keys())
    return sum(pA.get(it, 0.0) * pB.get(it, 0.0) for it in items)

def expected_rbo_at_k(items, human, other, k=10, p=0.9, ascending=True):
    """
    Expected RBO@k between human rank judgements and experimental ranks, tie-aware.

    - Builds tie-buckets from scores (ascending=True means smaller score = higher rank)
    - Uses expected overlap at each depth d=1..k
    - Computes truncated RBO@k: (1-p) * sum_{d=1..k} p^{d-1} * E[overlap(d)]/d
    """

    human_groups = tie_groups(human, items, ascending=ascending)
    other_groups = tie_groups(other, items, ascending=ascending)

    s = 0.0
    for d in range(1, k + 1):
        exp_olap = expected_overlap_at_depth(human_groups, other_groups, d)
        s += (exp_olap / d) * (p ** (d - 1))

    return (1 - p) * s


# Adapted from Clarke...
def rbo_clarke(run, ideal, depth, p):
    run_set = set()
    ideal_set = set()

    score = 0.0
    normalizer = 0.0
    weight = 1.0
    for i in range(depth):
        if i < len(run):
            run_set.add(run[i])
        if i < len(ideal):
            ideal_set.add(ideal[i])
        score += weight*len(ideal_set.intersection(run_set))/(i + 1)
        normalizer += weight
        weight *= p
    return score/normalizer

def compute_rbo_clause(human_ranks, other_ranks,p,skipFirst):
    idx = 0
    if skipFirst:
        idx = 1
    gold = sorted(zip(human_ranks['Doc'].tolist(),human_ranks['Rank'].tolist()),key = lambda x : (x[1],x[0]))
    gold_ranking = [d[0] for d in gold]
    #norm = rbo_clarke(gold_ranking,gold_ranking,len(gold_ranking),p)
    norm = expected_rbo_at_k(human_ranks['Doc'],gold_ranking,gold_ranking,len(gold_ranking),p)
    pred = sorted(zip(other_ranks['Doc'].tolist()[idx:],other_ranks['Rank'][idx:].tolist()),key = lambda x : (x[1],x[0]))
    pred_ranking = [d[0] for d in pred]
    #score = rbo_clarke(pred_ranking,gold_ranking,len(gold_ranking),p)
    score = expected_rbo_at_k(human_ranks['Doc'].tolist(),gold_ranking,pred_ranking,len(gold_ranking),p)
    
    return score/norm

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
        writer.writerow(['Clause','P','Value'])
        for  clause_type in clause_types:
            human_ranks = read_ranks(human_ranks_file_prefix,clause_type)
            other_ranks = read_ranks(other_ranks_file_prefix,clause_type)
            row = compute_rbo_clause(human_ranks, other_ranks,p,skipFirst)
            writer.writerow([clause_type,p,round(row,3)])
    