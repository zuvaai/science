import spacy
import glob

import pandas as pd

nlp = spacy.load("en_core_web_sm")

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

# create new dataframe to store sentence counts for all clause types
new_df = pd.DataFrame(columns=['Clause type', "Original", "Negated", "Rephrased"])

for clause_type in clause_types:
    num_sentences = []

    glob_path = f"{clause_dir}/Similarity Clauses - {clause_type}*.csv"
    filepath = glob.glob(glob_path)
    df = pd.read_csv(filepath[0])
    # take first 20 rows
    df = df.head(20)
    # count the number of sentences for each column
    for col_name in df.columns.tolist():
        sentence_count = 0
        for i, text in enumerate(df[col_name].tolist()):
            doc = nlp(text)
            sentences = [sent.text for sent in doc.sents]
            sentence_count = sentence_count+len(sentences)
        num_sentences.append(sentence_count)

    new_row = pd.Series([clause_type]+num_sentences, index=new_df.columns)
    new_df = pd.concat([new_df, pd.DataFrame([new_row])], ignore_index=True)

# sum the columns
new_df['Original'].sum()
new_row = pd.Series(["Total"]+[new_df['Original'].sum(), new_df['Negated'].sum(), new_df['Rephrased'].sum()], index=new_df.columns)
new_df = pd.concat([new_df, pd.DataFrame([new_row])], ignore_index=True)
# save to csv
new_df.to_csv("sentence_counts.csv", index=False)