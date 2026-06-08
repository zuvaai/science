#!/bin/bash


python3 calculate_rank_diff.py human/ ranked-clauses/ranked-gpt-5.2- analysis/ranked-5.2-rank_diff.csv

python3 calculate_rank_diff.py human/ ranked-clauses/ranked-gpt-4.1- analysis/ranked-4.1-rank_diff.csv

python3 calculate_rank_diff.py human/ sim-csvs/openai/text-embedding-3-large/ analysis/openai-rank_diff.csv skip
python3 calculate_score_diff.py sim-csvs/openai/text-embedding-3-large/ analysis/openai-score_diff.csv skip

python3 calculate_rank_diff.py human/ sim-csvs/gemini/gemini-embedding-001/ analysis/gemini-rank_diff.csv skip
python3 calculate_score_diff.py sim-csvs/gemini/gemini-embedding-001/ analysis/gemini-score_diff.csv skip

python3 calculate_rank_diff.py human/ aligned-clauses-one-stage/matched-gpt-5.2- analysis/aligned-one-5.2-rank_diff.csv skip
python3 calculate_score_diff.py aligned-clauses-one-stage/matched-gpt-5.2- analysis/aligned-one-5.2-score_diff.csv skip

python3 calculate_score_diff.py  aligned-clauses-one-stage/matched-gpt-4.1- analysis/aligned-one-4.1-score_diff.csv skip
python3 calculate_rank_diff.py human/ aligned-clauses-one-stage/matched-gpt-4.1- analysis/aligned-one-4.1-rank_diff.csv skip

python3 calculate_score_diff.py  aligned-clauses-two-stage/matched-gpt-5.2-gpt-5.2- analysis/aligned-two-5.2-score_diff.csv skip
python3 calculate_rank_diff.py human/ aligned-clauses-two-stage/matched-gpt-5.2-gpt-5.2- analysis/aligned-two-5.2-rank_diff.csv skip

python3 calculate_score_diff.py  aligned-clauses-two-stage/matched-gpt-4.1-gpt-4.1- analysis/aligned-two-4.1-score_diff.csv skip
python3 calculate_rank_diff.py human/ aligned-clauses-two-stage/matched-gpt-4.1-gpt-4.1- analysis/aligned-two-4.1-rank_diff.csv skip

python3 calculate_rank_diff.py human/ aligned-clauses-two-stage/matched-gpt-4.1-gpt-5.2- analysis/aligned-two-4.1-5.2-rank_diff.csv skip
python3 calculate_score_diff.py  aligned-clauses-two-stage/matched-gpt-4.1-gpt-5.2- analysis/aligned-two-4.1-5.2-score_diff.csv skip

python3 calculate_score_diff.py aligned-clauses-two-stage/matched-gpt-5.2-gpt-4.1- analysis/aligned-two-5.2-4.1-score_diff.csv skip
python3 calculate_rank_diff.py human/ aligned-clauses-two-stage/matched-gpt-5.2-gpt-4.1- analysis/aligned-two-5.2-4.1-rank_diff.csv skip