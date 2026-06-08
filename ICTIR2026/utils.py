
import os

# read list of csv files from directory
def get_csvs(dirpath: str):
    return [f for f in os.listdir(dirpath) if f.endswith('.csv')]

def strip_stuff(name: str):
    return name.replace(" ", "").replace(",", "").replace("/", "")

def strip_csv_path(file: str):
    return strip_stuff(file).replace("SimilarityClauses-", "").replace(".csv", "")