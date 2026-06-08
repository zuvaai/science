import pickle
import os

import numpy as np
import pandas as pd
from numpy import dot
from numpy.linalg import norm
from collections import defaultdict


def rank_clauses(clauses):
    scores = defaultdict(list)
    for clause in clauses:
        scores[clause[1]].append(clause)
    ranked = sorted(scores.items(), key =lambda tup: tup[0],reverse=True)
    results = []
    for i in range(len(ranked)):
        clause_score = ranked[i]
        for clause in clause_score[1]:
            clause=[clause[0],clause[1],i+1]
            results.append(clause)
    return results

def create_sim_csvs(clause_dir: str, experiment_path:str, clause_type:str, output_dir:str):

    # calculate the distances for the given embeddings and write to csv
    with open(f"{experiment_path}/{clause_type}/embeds.pkl", "rb") as f:
        docs,embeddings = pickle.load(f)

    # distances between pairs of sentences
    dists = []
    for idx,e1 in enumerate(embeddings):
        for e2 in embeddings:
            cos_sim = dot(np.asarray(e1),np.asarray(e2) )/(norm(e1)*norm(e2))
            dists.append(cos_sim)
        break
    ranks = [[docs[0],dists[0],-1]] # Template clause
    doc_sims = rank_clauses(zip(docs[1:],dists[1:]))
    ranks.extend(doc_sims)
    df = pd.DataFrame(ranks)
    df.columns = ['Doc','Score','Rank']
    output_dir = f"{output_dir}/{experiment_path}"
    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(f"{output_dir}/{clause_type}.csv", index = False)



if __name__ == "__main__":

    experiment_paths = [
        "openai/text-embedding-3-large",
        "gemini/gemini-embedding-001",
    ]

    clause_dir = "./clauses/"
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
    output_dir = "sim-csvs"

    for e in experiment_paths:
        for c in clause_types:
            create_sim_csvs(clause_dir, e, c, output_dir)

            