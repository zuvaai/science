

import pickle
import os
import glob

import numpy as np
import pandas as pd
from numpy import dot
from numpy.linalg import norm

def sigir_correctness_counts(sim_csv_dir, experiment_paths, clause_types):
    df = pd.DataFrame(columns=['Model', 'Correct L2', 'Incorrect L2', 'Correct Cos', 'Incorrect Cos'])
    for ex in experiment_paths:
        correct_l2 = 0 # desired behavior, same meaning closer than different
        incorrect_l2 = 0 # undesired behavior, different meaning closer than same

        correct_cos = 0 # as above
        incorrect_cos = 0
        for clause_type in clause_types:
            clause_df = pd.read_csv(f"{sim_csv_dir}/{ex}/{clause_type}/sim.csv")
            for i, row in clause_df.iterrows():
                if row['orig_vs_same_cos'] < row['orig_vs_diff_cos']:
                    incorrect_l2+=1
                else:
                    correct_l2+=1
                    print(f"{ex.split('/')[1]},  {clause_type}, example {i}, cosine")
                if row['orig_vs_same_L2'] > row['orig_vs_diff_L2']:
                    incorrect_cos+=1
                else:
                    correct_cos+=1
                    print(f"{ex.split('/')[1]},  {clause_type}, example {i}, L2")

    
        new_row = pd.DataFrame({'Model':ex.split('/')[1], 'Correct L2':[correct_l2], 'Incorrect L2':[incorrect_l2], 'Correct Cos':[correct_cos], 'Incorrect Cos':[incorrect_cos]})
        df = pd.concat([df, new_row], ignore_index=True)
        os.makedirs(f"{sim_csv_dir}/sigir-counts", exist_ok=True)
        df.to_csv(f"{sim_csv_dir}/sigir-counts/counts.csv", index=False)


def create_sentence_sim_csvs(clause_dir: str, experiment_path:str, clause_type:str, output_dir:str):

    glob_path = f"{clause_dir}/Similarity Clauses - {clause_type}*.csv"
    filepath = glob.glob(glob_path)
    df_text = pd.read_csv(filepath[0])
    df_text = df_text.dropna()

    if experiment_path.split('/')[1] == "legal-bert" and clause_type == "MFN":
        print("delete MFN row 14 for legal-bert") # HACK 
        df_text = df_text.drop(14)
        df_text = df_text.reset_index(drop=True)

    # calculate the distances for the given embeddings and write to csv
    with open(f"{experiment_path}/{clause_type}/Original.pkl", "rb") as f:
        original = pickle.load(f)
    with open(f"{experiment_path}/{clause_type}/SameMeaningWordedDifferently.pkl", "rb") as f:
        same_meaning = pickle.load(f)
    with open(f"{experiment_path}/{clause_type}/DifferentMeaningMinimalChanges.pkl", "rb") as f:
        different_meaning = pickle.load(f)

    # distances between original and same meaning
    orig_vs_same_cos = []
    orig_vs_same_L2 = []
    orig_vs_diff_cos = []
    orig_vs_diff_L2 = []
    for i in range(len(original)):

        original_sentences = original[i]
        same_meaning_sentences = same_meaning[i]
        different_meaning_sentences = different_meaning[i]

        orig_vs_same_cos_sentences = []
        orig_vs_same_L2_sentences = []
        orig_vs_diff_cos_sentences = []
        orig_vs_diff_L2_sentences = []

        # compute cosine similarity between all pairs of sentences
        for orig_sent in original_sentences:
            for same_sent in same_meaning_sentences:
                cos_sim = dot(np.asarray(orig_sent), np.asarray(same_sent)) / (norm(orig_sent) * norm(same_sent))
                orig_vs_same_cos_sentences.append(cos_sim)
                L2_sim = norm(np.asarray(orig_sent) - np.asarray(same_sent))
                orig_vs_same_L2_sentences.append(L2_sim)

        # min cos
        orig_vs_same_cos.append(min(orig_vs_same_cos_sentences))
        # max L2
        orig_vs_same_L2.append(max(orig_vs_same_L2_sentences))

        for orig_sent in original_sentences:
            for diff_sent in different_meaning_sentences:
                cos_sim = dot(np.asarray(orig_sent), np.asarray(diff_sent)) / (norm(orig_sent) * norm(diff_sent))
                orig_vs_diff_cos_sentences.append(cos_sim)
                L2_sim = norm(np.asarray(orig_sent) - np.asarray(diff_sent))
                orig_vs_diff_L2_sentences.append(L2_sim)

        # min cos
        orig_vs_diff_cos.append(min(orig_vs_diff_cos_sentences))
        # max L2
        orig_vs_diff_L2.append(max(orig_vs_diff_L2_sentences))

    # create dataframe
    df = pd.DataFrame({
        "orig_vs_same_cos": orig_vs_same_cos,
        "orig_vs_diff_cos": orig_vs_diff_cos,
        "orig_vs_same_L2": orig_vs_same_L2,
        "orig_vs_diff_L2": orig_vs_diff_L2,
    })

    output_dir = f"{output_dir}/{experiment_path}/{clause_type}"
    os.makedirs(output_dir, exist_ok=True)
    df.round(3).to_csv(f"{output_dir}/sim.csv", index = False)
    df.round(3).corr().to_csv(f"{output_dir}/corr.csv", index = False)

if __name__ == "__main__":

    experiment_paths = [
        "models/fasttext/crawl-300d-2M-subword-norm-sentences",

        "models/legal-bert/legal-bert-base-uncased-meanpool-norm-sentences",

        "models/nvidia/nv-embed-v2-norm-sentences",

        "models/openai/text-embedding-3-large-sentences",
        
        "models/gemini/text-embedding-004-sentences"
    ]
    clause_dir = "clauses"
    clause_types = [
        "Assignment",
        "Transfer of Data",
        "Exclusivity",
        "Non-Solicit",
        "Permitted Use of Data",
       "Audit Right",
        "License Grant",
        "MFN",
        "Publicity",
        "Termination for Convenience"
    ]

    output_dir = "sentence-sim-csvs"


    for e in experiment_paths:
        for c in clause_types:
            create_sentence_sim_csvs(clause_dir, e, c, output_dir)

    
    sigir_correctness_counts("sentence-sim-csvs", experiment_paths, clause_types)