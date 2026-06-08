
import os
import pickle
import glob

import pandas as pd
from openai import OpenAI
from tenacity import retry, wait_exponential
# import spacy

client = OpenAI(
     api_key=os.environ.get("OPENAI_API_KEY"),
)

@retry(wait=wait_exponential(multiplier=1,min=1,max=30))
def get_embedding(text: str, model: str):
    response = client.embeddings.create(
        input=text,
        model=model
    )
    usage = response.usage.prompt_tokens
    return response.data[0].embedding,usage


def embed_openai(clause_dir: str, experiment_name: str, clause_type:str, model: str):
    # each csv file contains a clause type
    print(f"Embedding {clause_type}...")
    # each csv file contains a clause type
    glob_path = f"{clause_dir}/{clause_type}.csv"
    filepath = glob.glob(glob_path)
    print(filepath)
    df = pd.read_csv(filepath[0])
    # embed all the text for each column and pickle
    ids = []
    usages = []
    for idx,col_name in enumerate(df.columns.tolist()):
        embeddings = []
        if idx == 0:
            ids = df[col_name].tolist()
            continue
        for i, text in enumerate(df[col_name].tolist()):
            print(f"Embedding {col_name} {i}")
            emb,usage = get_embedding(text, model)
            usages.append(usage)
            embeddings.append(emb)
        output_dir = f"{experiment_name}/{clause_type}"
        os.makedirs(output_dir, exist_ok=True)
        with open(f"{output_dir}/usage.csv", 'w') as w:
            for usage in usages:
                w.write(f"{usage}\n")
        with open(f"{output_dir}/embeds.pkl", 'wb') as w:
            print(f"Pickling {col_name}")
            pickle.dump([ids,embeddings], w)

if __name__ == "__main__":

    # experiment_name is used to distinguish between different embedding models or configurations,
    # used to name output directory

    experiment_name = "openai/text-embedding-3-large" 
    model = "text-embedding-3-large"

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
    for clause_type in clause_types:
        embed_openai(clause_dir, experiment_name, clause_type, model)