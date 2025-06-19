import os
import pickle
import glob

from sentence_transformers import SentenceTransformer
import pandas as pd
import utils

import spacy


def add_eos(input_examples):
  input_examples = [input_example + model.tokenizer.eos_token for input_example in input_examples]
  return input_examples

def add_eos_single_string(input_string):
  return input_string + model.tokenizer.eos_token

def embed_nvidia(clause_dir:str, experiment_name: str,clause_type:str, normalize_embeddings: bool):
  # load model with tokenizer
  model.max_seq_length = 32768
  model.tokenizer.padding_side="right"
  batch_size = 1
  
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
          emb = model.encode(add_eos_single_string(text), batch_size=batch_size, normalize_embeddings=normalize_embeddings)
          embeddings.append(emb.tolist())
      output_dir = f"{experiment_name}/{clause_type}"
      os.makedirs(output_dir, exist_ok=True)
      with open(f"{output_dir}/{utils.strip_stuff(col_name)}.pkl", 'wb') as w:
          print(f"Pickling {col_name}")
          pickle.dump(embeddings, w)

def embed_nvidia_sentences(clause_dir: str, experiment_name: str, clause_type: str, normalize_embeddings: bool):
    # load model with tokenizer
    model.max_seq_length = 32768
    model.tokenizer.padding_side = "right"    
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
            
            # If no sentences were found, use the original text
            if not sentences:
                sentences = [text]
            
            # Add EOS token to all sentences
            sentences_with_eos = [add_eos_single_string(sent) for sent in sentences]
            batch_size = len(sentences_with_eos)  # Set batch size to the number of sentences
            # Batch encode all sentences at once
            sentence_embeddings = model.encode(
                sentences_with_eos, 
                batch_size=batch_size, 
                normalize_embeddings=normalize_embeddings
            ).tolist()
            
            # Store the embeddings for all sentences in this text
            embeddings.append(sentence_embeddings)
            
        output_dir = f"{experiment_name}/{clause_type}"
        os.makedirs(output_dir, exist_ok=True)
        with open(f"{output_dir}/{utils.strip_stuff(col_name)}.pkl", 'wb') as w:
            print(f"Pickling {col_name}")
            pickle.dump(embeddings, w)


if __name__ == "__main__":

    # experiment_name is used to distinguish between different embedding models or configurations,
    # used to name output directory
    experiment_name = "nvidia/nv-embed-v2-norm-sentences" 
    normalize_embeddings = True

    model = SentenceTransformer('nvidia/NV-Embed-v2', trust_remote_code=True)

    clause_dir = "../clauses"
    clause_types = [
        # "Assignment",
        # "Transfer of Data",
        # "Exclusivity",
        # "Non-Solicit",
        # "Permitted Use of Data",
        "Audit Right",
        "License Grant",
        "MFN",
        "Publicity",
        "Termination for Convenience"
    ]

    # for clause_type in clause_types:
    #     embed_nvidia(clause_dir, experiment_name, clause_type, normalize_embeddings)

    nlp = spacy.load("en_core_web_sm")

    for clause_type in clause_types:
        embed_nvidia_sentences(clause_dir, experiment_name, clause_type, normalize_embeddings)