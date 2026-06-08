#!/bin/bash

python3 calculate_ami.py human/ ranked-clauses/ranked-gpt-5.2- results/ranked-5.2-ami.csv
python3 calculate_rbo.py human/ ranked-clauses/ranked-gpt-5.2- results/ranked-5.2-rbo.csv

python3 calculate_ami.py human/ ranked-clauses/ranked-gpt-4.1- results/ranked-4.1-ami.csv
python3 calculate_rbo.py human/ ranked-clauses/ranked-gpt-4.1- results/ranked-4.1-rbo.csv

python3 calculate_ami.py human/ sim-csvs/openai/text-embedding-3-large/ results/openai-ami.csv skip
python3 calculate_rbo.py human/ sim-csvs/openai/text-embedding-3-large/ results/openai-rbo.csv skip

python3 calculate_ami.py human/ sim-csvs/gemini/gemini-embedding-001/ results/gemini-ami.csv skip
python3 calculate_rbo.py human/ sim-csvs/gemini/gemini-embedding-001/ results/gemini-rbo.csv skip

python3 calculate_ami.py human/ aligned-clauses-one-stage/matched-gpt-5.2- results/aligned-one-5.2-ami.csv skip
python3 calculate_rbo.py human/ aligned-clauses-one-stage/matched-gpt-5.2- results/aligned-one-5.2-rbo.csv skip

python3 calculate_rbo.py human/ aligned-clauses-one-stage/matched-gpt-4.1- results/aligned-one-4.1-rbo.csv skip
python3 calculate_ami.py human/ aligned-clauses-one-stage/matched-gpt-4.1- results/aligned-one-4.1-ami.csv skip

python3 calculate_rbo.py human/ aligned-clauses-two-stage/matched-gpt-5.2-gpt-5.2- results/aligned-two-5.2-rbo.csv skip
python3 calculate_ami.py human/ aligned-clauses-two-stage/matched-gpt-5.2-gpt-5.2- results/aligned-two-5.2-ami.csv skip

python3 calculate_rbo.py human/ aligned-clauses-two-stage/matched-gpt-4.1-gpt-4.1- results/aligned-two-4.1-rbo.csv skip
python3 calculate_ami.py human/ aligned-clauses-two-stage/matched-gpt-4.1-gpt-4.1- results/aligned-two-4.1-ami.csv skip

python3 calculate_ami.py human/ aligned-clauses-two-stage/matched-gpt-4.1-gpt-5.2- results/aligned-two-4.1-5.2-ami.csv skip
python3 calculate_rbo.py human/ aligned-clauses-two-stage/matched-gpt-4.1-gpt-5.2- results/aligned-two-4.1-5.2-rbo.csv skip

python3 calculate_rbo.py human/ aligned-clauses-two-stage/matched-gpt-5.2-gpt-4.1- results/aligned-two-5.2-4.1-rbo.csv skip
python3 calculate_ami.py human/ aligned-clauses-two-stage/matched-gpt-5.2-gpt-4.1- results/aligned-two-5.2-4.1-ami.csv skip