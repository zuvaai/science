# Explainable Legal Similarity

The main data for the experiments conducted in this paper.

`clauses` contains the raw clauses to be compared including the reference clause.

`human` contains the exported human judgements (including any notes).

The main results can be re-produced using the `rank_clauses.sh`,`compute_embedding_similarity.sh`, and `parse_clauses.sh` followed by `align_clauses.sh`.

This will overwrite the results present in `sim-csvs` (Embedding results), `ranked-clauses` (Listwise results), `aligned-clauses-one-stage` (One-Prompt), and `aligned-clauses-two-stage` (Two-Prompt). 

Main evaluation can be conducted by running `evaluate.sh` (and results in `results`). The additional analysis of score and rank differences are calculated using `additional_analysis.sh` (and stored in `analysis`).

The data for the brief qualitative review of the outputs by our lawyer can be found in `pilot`.

Prompts for the various methods can be found in `obligation_parsing.py`/`obligation_matching.py` (Two-Prompt), `obligation_alignment.py` (One-prompt), and `obligation_ranking.py` (Listwise).