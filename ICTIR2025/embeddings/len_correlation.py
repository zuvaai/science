
import pandas as pd

import os
import glob

def length_correlations(clause_dir, sim_dir, output_dir, experiment_paths, clause_type, sim_types):
    # create csv with correlations between the length of the original clause and the similarity values
    os.makedirs(output_dir, exist_ok=True)
    glob_path = f"{clause_dir}/Similarity Clauses - {clause_type}*.csv"
    filepath = glob.glob(glob_path)
    print(filepath)
    df_text = pd.read_csv(filepath[0])
    len_correlations = []
    for ex in experiment_paths:
        if ex.split('/')[1] == "legal-bert" and clause_type == "MFN":
            print("delete MFN row 14 for legal-bert") # HACK, bert must be last in experiment_paths list!
            df_text = df_text.drop(14)
            df_text = df_text.reset_index(drop=True)
        for sim_type in sim_types:
            df_sim = pd.read_csv(f"{sim_dir}/{ex}/{clause_type}/sim.csv")
            orig_vs_diff = df_sim[f"orig_vs_diff_{sim_type}"]
            orig_len = df_text['Original'].str.len()
            len_correlations.append({
                "model": ex,
                "sim_type": sim_type,
                "pearson": orig_len.corr(orig_vs_diff)
            })
    df_pearson = pd.DataFrame(len_correlations)
    output= f"{output_dir}/{clause_type}/len_correlations.csv"
    os.makedirs(os.path.dirname(output), exist_ok=True)
    # write to csv but don't keep the index
    df_pearson.to_csv(output, index = False)

if __name__ == "__main__":

    sim_dir = "sim-csvs" # directory containing the csv files with the similarity values

    experiment_paths = [
            "models/fasttext/crawl-300d-2M-subword-norm",
            
            "models/nvidia/nv-embed-v2-norm",

            "models/openai/text-embedding-3-large",
            
            "models/gemini/text-embedding-004",

            "models/legal-bert/legal-bert-base-uncased-meanpool-norm", # must be last for HACK to work

        ]
    
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
    

    output_dir = "length-correlations"
    clause_dir = "clauses" # contains csv files with the clause text
    sim_types = ["cos", "L2"]
    for clause_type in clause_types:
        length_correlations(clause_dir, sim_dir, output_dir, experiment_paths, clause_type, sim_types)







