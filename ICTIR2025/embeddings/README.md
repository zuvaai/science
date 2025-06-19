

# Embeddings

## To use

Use python3.10, if you don't have it get it with `brew install python@3.10`

Create a virtual env with `python3.10 -m venv ve` and activate it `source ./ve/bin/activate`

Install the dependencies with `python -m pip install -r requirements.txt`


There is a script per model in the `models/` dir for getting the embeddings.

`similarity.py` computes various similarity measures.

`plots.py` produces some plots.

`len_correlation.py` produces some csvs showing the correlation between clause length and similarity.