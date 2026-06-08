#!/bin/bash

#Does the initial embedding and then ranking of clauses using OpenAI 'text-embedding-3-large' model and Gemini 'gemini-embedding-001' with the task set to semantic similarity

python3 open_ai.py
python3 gemini.py
python3 similarity_ranked.py
