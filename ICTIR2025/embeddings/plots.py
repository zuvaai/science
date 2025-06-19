
import os
import glob

import matplotlib.pyplot as plt
import pandas as pd

def lineplot(sim_csv_dir:str, experiment_path:str, clause_type:str, output_dir:str):
    # read the csv file
    df = pd.read_csv(f"{sim_csv_dir}/{experiment_path}/{clause_type}/sim.csv")
    # plot the lines
    _, ax = plt.subplots()
    ax.set_title(f"{experiment_path}")
    # ax.plot(df['orig_vs_same_cos'].tolist(), label="Original vs Same Meaning cos", color="green", marker="o")
    # ax.plot(df['orig_vs_diff_cos'].tolist(), label="Original vs Different Meaning cos", color="red", marker="o")
    ax.plot(df['orig_vs_same_L2'].tolist(), label="Original vs Same Meaning L2", color="green")
    ax.plot(df['orig_vs_diff_L2'].tolist(), label="Original vs Different Meaning L2", color="red")
    ax.set_xlabel("Clause Number")
    ax.set_ylabel("Similarity")
    ax.legend()
    ax.set_xticks(range(len(df['orig_vs_same_cos'].tolist())))
    ax.set_xticklabels(range(1, len(df['orig_vs_same_cos'].tolist()) + 1))
    ax.grid()
    output_path = f"{output_dir}/{experiment_path}/{clause_type}/lineplot.png"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)

def box_and_whisker(sim_csv_dir:str, experiment_path:str, clause_type:str, output_dir:str, sim_type:str):
    # read the csv file
    df = pd.read_csv(f"{sim_csv_dir}/{experiment_path}/{clause_type}/sim.csv")
    # plot the lines
    _, ax = plt.subplots()
    ax.set_title(f"{experiment_path}-{sim_type}")
    if sim_type == "cos":
        ax.boxplot([df['orig_vs_same_cos'].tolist(), df['orig_vs_diff_cos'].tolist()])
        ax.set_xticklabels(["origVsame", "origVdiff"])
    elif sim_type == "L2":
        ax.boxplot([df['orig_vs_same_L2'].tolist(), df['orig_vs_diff_L2'].tolist()])
        ax.set_xticklabels(["origVsame", "origVdiff"])
    else:
        raise ValueError("sim_type must be 'cos' or 'L2'")
    ax.set_ylabel("Similarity")
    ax.grid()
    output_path = f"{output_dir}/{experiment_path}/boxplot-{sim_type}.png"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)

def scatter_plot(clause_dir:str, sim_csv_dir: str, experiment_path: str, clause_type: str, output_dir: str, sim_type: str):
    # read csv file with text to get lengths
    glob_path = f"{clause_dir}/Similarity Clauses - {clause_type}*.csv"
    filepath = glob.glob(glob_path)
    df_text = pd.read_csv(filepath[0])
    df_text = df_text.dropna()
    if experiment_path.split('/')[1] == "legal-bert" and clause_type == "MFN": #HACK!
        print("delete MFN row 14 for legal-bert") 
        df_text = df_text.drop(14)
        df_text = df_text.reset_index(drop=True)
    # read the csv file with sim values
    df_sim = pd.read_csv(f"{sim_csv_dir}/{experiment_path}/{clause_type}/sim.csv")
    # plot the scatter
    _, ax = plt.subplots()
    ax.set_title(f"{experiment_path}")
    ax.scatter(df_text['Original'].str.len(), df_sim[f'orig_vs_diff_{sim_type}'].tolist())
    ax.set_xlabel("Original Char Length")
    ax.set_ylabel(f"{sim_type}")
    ax.grid()
    output = f"{output_dir}/{experiment_path}/{clause_type}/char-len-vs-{sim_type}.png"
    os.makedirs(os.path.dirname(output), exist_ok=True)
    plt.savefig(output)

def sigir_boxplots(sim_csv_dir, experiment_paths, clause_types, output_dir):
     # sigir paper plots
    # output_dir = "plots/sigir-paper"

    sim_types = ["cos", "L2"]
    for sim_type in sim_types:
        fig, ax = plt.subplots()
        if sim_type == "cos":
            ax.set_ylabel("Similarity")

        if sim_type == "L2":
            ax.set_ylabel("Distance")

        data = []
        labels = []
        positions = []
        pos = 1
        for experiment_path in experiment_paths:
            pair_name = experiment_path.split('/')[1]
            same_meaning = []
            diff_meaning = []
            for clause_type in clause_types:
                df = pd.read_csv(f"{sim_csv_dir}/{experiment_path}/{clause_type}/sim.csv")
                same_meaning.extend(df[f"orig_vs_same_{sim_type}"].tolist())
                diff_meaning.extend(df[f"orig_vs_diff_{sim_type}"].tolist())
            data.append(same_meaning)
            data.append(diff_meaning)
            labels.append(f"{pair_name}_same")
            labels.append(f"{pair_name}_diff")
            positions.extend([pos, pos + 1])
            pos += 3  # Add space between pairs

        bplot=ax.boxplot(data, positions=positions, patch_artist=True, flierprops={'marker': '.', 'markersize': 2}, medianprops={'color': 'black'})
        # fill in pairs with colors
        boxes = bplot['boxes']
        # fill boxes in pairs
        for i in range(0, len(boxes), 2):
            box1 = boxes[i]
            box2 = boxes[i + 1]
            box1.set_facecolor('#1A85FF') # blue good
            box2.set_facecolor('#D41159') # red bad
            # set aplpha
            box1.set_alpha(0.5)
            box2.set_alpha(0.5)

        ax.set_xticks([pos -1 for pos in positions[::2]])
        ax.set_xticklabels([label.split('_')[0] for label in labels[::2]],ha='left', rotation=45, fontsize=10, fontweight='bold')
        ax.legend([box1, box2], ['Rephrased', 'Negation'])
        ax.grid()
        plt.tight_layout()
        output_path = f"{output_dir}/boxplot-{sim_type}.png"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path)

if __name__ == "__main__":

    # sim_csv_dir = "sim-csvs"
  
    # experiment_paths = [
    #     "models/fasttext/crawl-300d-2M-subword-norm",

    #     "models/legal-bert/legal-bert-base-uncased-meanpool-norm",

    #     "models/nvidia/nv-embed-v2-norm",

    #     "models/openai/text-embedding-3-large",
        
    #     "models/gemini/text-embedding-004"
    # ]


    sim_csv_dir = "sentence-sim-csvs"
    # experiment_paths = [
    #     "models/fasttext/crawl-300d-2M-subword-norm-sentences",
    #     "models/legal-bert/legal-bert-base-uncased-meanpool-norm-sentences",
    # ]

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

    # sigir_boxplots(sim_csv_dir, experiment_paths, clause_types, "plots/sigir-paper")

    # for experiment_path in experiment_paths:
    #     for clause_type in clause_types:
    #         lineplot(sim_csv_dir, experiment_path, clause_type, "plots/sentence-lineplots")


    
    experiment_paths = [
        "models/fasttext/crawl-300d-2M-subword-norm-sentences",

        "models/legal-bert/legal-bert-base-uncased-meanpool-norm-sentences",

        "models/nvidia/nv-embed-v2-norm-sentences",

        "models/openai/text-embedding-3-large-sentences",
        
        "models/gemini/text-embedding-004-sentences"
    ]

    sigir_boxplots(sim_csv_dir, experiment_paths, clause_types, "plots/sigir-paper-sentences")
