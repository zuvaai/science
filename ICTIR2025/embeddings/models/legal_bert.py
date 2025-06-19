
import os
import pickle
import glob

import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModel
import spacy

import utils


def embed_legal_bert(clause_dir: str, experiment_name: str, clause_type: str, normalize_embeddings: bool):
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
            if clause_type == "MFN" and i == 14: # HACK too long for BERT
                print("Skipping MFN 14")
                continue
            encoded_input = tokenizer(text, return_tensors='pt')
            output = model(**encoded_input)

            # commented line below uses the CLS token
            # emb = output.pooler_output.tolist()[0]

            # use the mean of the hidden states, discarding the CLS and SEP tokens
            states = output['last_hidden_state'][0][1:-1]
            emb = states.mean(dim=0).tolist()
            # normalize
            if normalize_embeddings:
                emb =  emb / np.linalg.norm(emb)
            embeddings.append(emb)

        output_dir = f"{experiment_name}/{clause_type}"
        os.makedirs(output_dir, exist_ok=True)
        with open(f"{output_dir}/{utils.strip_stuff(col_name)}.pkl", 'wb') as w:
            print(f"Pickling {col_name}")
            pickle.dump(embeddings, w)  


def embed_legal_bert_sentences(clause_dir: str, experiment_name: str, clause_type: str, normalize_embeddings: bool):
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
            if clause_type == "MFN" and i == 14: # HACK too long for BERT
                print("Skipping MFN 14")
                continue
            # split text into sentences
            doc = nlp(text)
            sentences = [sent.text for sent in doc.sents]
            sentence_embeddings = []
            for sent in sentences:
                encoded_input = tokenizer(sent, return_tensors='pt')
                output = model(**encoded_input)

                # commented line below uses the CLS token
                # emb = output.pooler_output.tolist()[0]

                # use the mean of the hidden states, discarding the CLS and SEP tokens
                states = output['last_hidden_state'][0][1:-1]
                emb = states.mean(dim=0).tolist()
                # normalize
                if normalize_embeddings:
                    emb =  emb / np.linalg.norm(emb)
                sentence_embeddings.append(emb)

            embeddings.append(sentence_embeddings)
        output_dir = f"{experiment_name}/{clause_type}"
        os.makedirs(output_dir, exist_ok=True)
        with open(f"{output_dir}/{utils.strip_stuff(col_name)}.pkl", 'wb') as w:
            print(f"Pickling {col_name}")
            pickle.dump(embeddings, w)  

if __name__ == "__main__":

    # experiment_name is used to distinguish between different embedding models or configurations,
    # used to name output directory
    experiment_name = "legal-bert/legal-bert-base-uncased-meanpool-norm-sentences" 
    normalize_embeddings = True
    
    tokenizer = AutoTokenizer.from_pretrained("nlpaueb/legal-bert-base-uncased")
    model = AutoModel.from_pretrained("nlpaueb/legal-bert-base-uncased")
    
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


    # for clause_type in clause_types:
    #     embed_legal_bert(clause_dir, experiment_name, clause_type, normalize_embeddings)

    nlp = spacy.load("en_core_web_sm")

    # embed individual sentences
    for clause_type in clause_types:
        embed_legal_bert_sentences(clause_dir, experiment_name, clause_type, normalize_embeddings)
