
import os
import pickle
import glob

import pandas as pd
from openai import OpenAI
from tenacity import retry, wait_exponential
import spacy

import utils

client = OpenAI(
     api_key=os.environ.get("OPENAI_API_KEY"),
)

@retry(wait=wait_exponential(multiplier=1,min=1,max=30))
def get_embedding(text: str, model: str):
    response = client.embeddings.create(
        input=text,
        model=model
    )
    return response.data[0].embedding


def embed_openai(clause_dir: str, experiment_name: str, clause_type:str, model: str):
    # each csv file contains a clause type
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
            emb = get_embedding(text, model)
            embeddings.append(emb)
        output_dir = f"{experiment_name}/{clause_type}"
        os.makedirs(output_dir, exist_ok=True)
        with open(f"{output_dir}/{utils.strip_stuff(col_name)}.pkl", 'wb') as w:
            print(f"Pickling {col_name}")
            pickle.dump(embeddings, w)

def embed_openai_sentences(clause_dir: str, experiment_name: str, clause_type:str, model: str):
    # each csv file contains a clause type
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
            doc = nlp(text)
            sentences = [sent.text for sent in doc.sents]
            sentence_embeddings = []
            for sent in sentences:
                emb = get_embedding(sent, model)
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
    experiment_name = "openai/text-embedding-3-large-sentences" 
    model = "text-embedding-3-large"  # either "text-embedding-3-large" or "text-embedding-3-small"

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
    #     embed_openai(clause_dir, experiment_name, clause_type, model)

    nlp = spacy.load("en_core_web_sm")

    # embed individual sentences
    for clause_type in clause_types:
        embed_openai_sentences(clause_dir, experiment_name, clause_type, model)