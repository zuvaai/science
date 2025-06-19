
import os
import pickle
import glob

import fasttext
import pandas as pd
import numpy as np
import spacy

import utils


def embed_fasttext(clause_dir: str, experiment_name: str, clause_type: str, normalize_embeddings: bool):
    print(f"Embedding {clause_type}...")
    # each csv file contains a clause type
    glob_path = f"{clause_dir}/Similarity Clauses - {clause_type}*.csv"
    filepath = glob.glob(glob_path)
    print(filepath)
    df = pd.read_csv(filepath[0])
    # take first 20 rows
    df = df.head(20)
    # embed all the text for each column and pickle
    for col_name in df.columns.tolist():
        embeddings = []
        for i, text in enumerate(df[col_name].tolist()):
            print(f"Embedding {col_name} {i}")
            text = text.replace("\n", " ")
            emb = model.get_sentence_vector(text)
            # normalize
            if normalize_embeddings:
                emb =  emb / np.linalg.norm(emb)
            embeddings.append(emb.tolist())
        output_dir = f"{experiment_name}/{clause_type}"
        os.makedirs(output_dir, exist_ok=True)
        with open(f"{output_dir}/{utils.strip_stuff(col_name)}.pkl", 'wb') as w:
            print(f"Pickling {col_name}")
            pickle.dump(embeddings, w)

def embed_fasttext_sentences(clause_dir: str, experiment_name: str, clause_type: str, normalize_embeddings: bool):
    print(f"Embedding {clause_type}...")
    # each csv file contains a clause type
    glob_path = f"{clause_dir}/Similarity Clauses - {clause_type}*.csv"
    filepath = glob.glob(glob_path)
    print(filepath)
    df = pd.read_csv(filepath[0])
    # take first 20 rows
    df = df.head(20)
    # embed all the text for each column and pickle
    for col_name in df.columns.tolist():
        embeddings = []
        for i, text in enumerate(df[col_name].tolist()):
            print(f"Embedding {col_name} {i}")
            text = text.replace("\n", " ")
            doc = nlp(text)
            sentences = [sent.text for sent in doc.sents]
            sentence_embeddings = []
            for sent in sentences:
                emb = model.get_sentence_vector(sent)
                # normalize
                if normalize_embeddings:
                    emb =  emb / np.linalg.norm(emb)
                sentence_embeddings.append(emb.tolist())
            embeddings.append(sentence_embeddings)
        output_dir = f"{experiment_name}/{clause_type}"
        os.makedirs(output_dir, exist_ok=True)
        with open(f"{output_dir}/{utils.strip_stuff(col_name)}.pkl", 'wb') as w:
            print(f"Pickling {col_name}")
            pickle.dump(embeddings, w)

if __name__ == "__main__":

    # experiment_name is used to distinguish between different embedding models or configurations,
    # used to name output directory
    experiment_name = "fasttext/crawl-300d-2M-subword-norm-sentences" # SET THIS CAREFULLY
    normalize_embeddings = True

    clause_dir = "../clauses"
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

    model = fasttext.load_model("fasttext-models/crawl-300d-2M-subword/crawl-300d-2M-subword.bin")

    # for clause_type in clause_types:
    #     embed_fasttext(clause_dir, experiment_name, clause_type, normalize_embeddings)

    nlp = spacy.load("en_core_web_sm")

    # embed individual sentences
    for clause_type in clause_types:
        embed_fasttext_sentences(clause_dir, experiment_name, clause_type, normalize_embeddings)