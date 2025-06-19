import pickle
import os
import glob

import numpy as np
import pandas as pd
from numpy import dot
from numpy.linalg import norm
import nltk
from spacy.tokenizer import Tokenizer
from spacy.lang.en import English


def bow_diff(a:str, b:str):
    a_toks = tokenizer(a)
    b_toks = tokenizer(b)
    # calculate the bag of words overlap
    a_bow = [tok.text for tok in a_toks]
    b_bow = [tok.text for tok in b_toks]
    overlap = len(set(a_bow).intersection(set(b_bow))) / len(set(a_bow).union(set(b_bow)))
    return overlap

def create_sim_csvs(clause_dir: str, experiment_path:str, clause_type:str, output_dir:str):

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
    orig_vs_same_edit = []
    orig_vs_same_bow = []
    for i in range(len(original)):
        cos_sim = dot(np.asarray(original[i]),np.asarray(same_meaning[i]) )/(norm(original[i])*norm(same_meaning[i]))
        L2_sim = norm(np.asarray(original[i]) - np.asarray(same_meaning[i]))
        edit_sim = nltk.edit_distance(df_text.iloc[i]['Original'], df_text.iloc[i]['Same Meaning, Worded Differently'])
        bow_sim = bow_diff(df_text.iloc[i]['Original'], df_text.iloc[i]['Same Meaning, Worded Differently'])
        orig_vs_same_cos.append(cos_sim)
        orig_vs_same_L2.append(L2_sim)
        orig_vs_same_edit.append(edit_sim)
        orig_vs_same_bow.append(bow_sim)

    # distances between original and different meaning
    orig_vs_diff_cos = []
    orig_vs_diff_L2 = []
    orig_vs_diff_edit = []
    orig_vs_diff_bow = []
    for i in range(len(original)):
        cos_sim = dot(original[i],different_meaning[i] )/(norm(original[i])*norm(different_meaning[i]))
        L2_sim = norm(np.asarray(original[i]) - np.asarray(different_meaning[i]))
        edit_sim = nltk.edit_distance(df_text.iloc[i]['Original'], df_text.iloc[i]['Different Meaning, Minimal Changes'])
        bow_sim = bow_diff(df_text.iloc[i]['Original'], df_text.iloc[i]['Different Meaning, Minimal Changes'])
        orig_vs_diff_cos.append(cos_sim)
        orig_vs_diff_L2.append(L2_sim)
        orig_vs_diff_edit.append(edit_sim)
        orig_vs_diff_bow.append(bow_sim)

    # create dataframe
    df = pd.DataFrame({
        "orig_vs_same_cos": orig_vs_same_cos,
        "orig_vs_diff_cos": orig_vs_diff_cos,
        "orig_vs_same_L2": orig_vs_same_L2,
        "orig_vs_diff_L2": orig_vs_diff_L2,
        "orig_vs_same_edit": orig_vs_same_edit,
        "orig_vs_diff_edit": orig_vs_diff_edit,
        "orig_vs_same_bow": orig_vs_same_bow,
        "orig_vs_diff_bow": orig_vs_diff_bow
    })

    output_dir = f"{output_dir}/{experiment_path}/{clause_type}"
    os.makedirs(output_dir, exist_ok=True)
    df.round(3).to_csv(f"{output_dir}/sim.csv", index = False)
    df.round(3).corr().to_csv(f"{output_dir}/corr.csv", index = False)


def sigir_correlation_data(sim_csv_dir, experiment_paths, clause_types):
    # generate a number for each experiment with metric
    for ex in experiment_paths:
        df = pd.DataFrame()
        for clause_type in clause_types:
            glob_path = f"{clause_dir}/Similarity Clauses - {clause_type}*.csv"
            filepath = glob.glob(glob_path)
            print(filepath)
            df_text = pd.read_csv(filepath[0])
            if ex.split('/')[1] == "legal-bert" and clause_type == "MFN":
                print("delete MFN row 14 for legal-bert") # HACK, bert must be last in experiment_paths list!
                df_text = df_text.drop(14)
                df_text = df_text.reset_index(drop=True)
            df_text = df_text.dropna()
            clause_df = pd.read_csv(f"{sim_csv_dir}/{ex}/{clause_type}/sim.csv")
            clause_df['Original Len'] = df_text['Original'].str.len()
            df = pd.concat([df, clause_df], axis=0)

        corr = df.corr()
        os.makedirs(f"{sim_csv_dir}/sigir-correlations", exist_ok=True)
        corr.round(3).to_csv(f"{sim_csv_dir}/sigir-correlations/{ex.split('/')[1]}.csv", index = True)

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

if __name__ == "__main__":

    experiment_paths = [
        "models/fasttext/crawl-300d-2M-subword-norm",

        "models/legal-bert/legal-bert-base-uncased-meanpool-norm",

        "models/nvidia/nv-embed-v2-norm",

        "models/openai/text-embedding-3-large",
        
        "models/gemini/text-embedding-004"
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

    output_dir = "sim-csvs"

    # use spacy tokenizer to calulate bow overlap
    nlp = English()
    tokenizer = Tokenizer(nlp.vocab)

    for e in experiment_paths:
        for c in clause_types:
            create_sim_csvs(clause_dir, e, c, output_dir)

    sim_csv_dir = "sim-csvs"
    sigir_correlation_data(sim_csv_dir, experiment_paths, clause_types)


    sigir_correctness_counts(sim_csv_dir, experiment_paths, clause_types)
            