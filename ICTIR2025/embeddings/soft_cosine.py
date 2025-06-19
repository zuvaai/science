

import logging
import glob
import os
import pickle

import pandas as pd
from nltk.corpus import stopwords
from nltk import download
from gensim.corpora import Dictionary
from gensim.models import TfidfModel
import gensim.downloader as api
from gensim.similarities import SparseTermSimilarityMatrix, WordEmbeddingSimilarityIndex
import matplotlib.pyplot as plt

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

def preprocess(sentence):
    return [w for w in sentence.lower().split() if w not in stop_words]

def cosine_similarity(clause_type:str, experiment_name:str, output_dir:str):
    glob_path = f"{clause_dir}/Similarity Clauses - {clause_type}*.csv"
    filepath = glob.glob(glob_path)
    print(filepath)
    df = pd.read_csv(filepath[0])
    df = df.head(20)
    original_list = df['Original'].tolist()
    same_list = df['Same Meaning, Worded Differently'].tolist()
    diff_list = df['Different Meaning, Minimal Changes'].tolist()

    original_list = [preprocess(sentence) for sentence in original_list]
    same_list = [preprocess(sentence) for sentence in same_list]
    diff_list = [preprocess(sentence) for sentence in diff_list]

    dictionary = Dictionary(original_list + same_list + diff_list)

    original_list = [dictionary.doc2bow(doc) for doc in original_list]
    same_list = [dictionary.doc2bow(doc) for doc in same_list]
    diff_list = [dictionary.doc2bow(doc) for doc in diff_list]

    tfidf = TfidfModel(original_list + same_list + diff_list)

    original_list = [tfidf[doc] for doc in original_list]
    same_list = [tfidf[doc] for doc in same_list]
    diff_list = [tfidf[doc] for doc in diff_list]

    termsim_index = WordEmbeddingSimilarityIndex(model)
    termsim_matrix = SparseTermSimilarityMatrix(termsim_index, dictionary, tfidf)

    orig_vs_same = []
    for i in range(len(original_list)):
        similarity = termsim_matrix.inner_product(original_list[i], same_list[i], normalized=(True, True))
        orig_vs_same.append(similarity)

    orig_vs_diff = []
    for i in range(len(original_list)):
        similarity = termsim_matrix.inner_product(original_list[i], diff_list[i], normalized=(True, True))
        orig_vs_diff.append(similarity)

    output_dir = f"{output_dir}/{experiment_name}/{clause_type}"
    os.makedirs(output_dir, exist_ok=True)

    _, ax = plt.subplots()
    ax.set_title(f"{clause_type} - Soft Cosine Similarity")
    ax.plot(orig_vs_same, label="Original vs Same Meaning", color="green", marker="o")
    ax.plot(orig_vs_diff, label="Original vs Different Meaning", color="red", marker="o")
    ax.set_xlabel("Clause Number")
    ax.set_ylabel("Similarity")
    ax.legend()
    ax.set_xticks(range(len(orig_vs_same)))
    ax.set_xticklabels(range(1, len(orig_vs_same) + 1))
    ax.grid()
    plt.savefig(f"{output_dir}/soft_cosine_similarity.png")
    plt.close()


if __name__ == "__main__":

    # experiment_name is used to distinguish between different embedding models or configurations,
    # used to name output directory
    experiment_name = "word2vec-google-news-300/" # SET THIS CAREFULLY
    output_dir = f"soft-cosine-plots/"

    # global variables
    download('stopwords')  
    stop_words = stopwords.words('english')
    model = api.load('word2vec-google-news-300')

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

    for clause_type in clause_types:
        cosine_similarity(clause_type, experiment_name, output_dir)



