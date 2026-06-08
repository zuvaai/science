#!/bin/bash


mkdir -p ranked-clauses
for i in clauses/*;
do
  python3 obligation_ranking.py $i ranked-clauses gpt-4.1
  python3 obligation_ranking.py $i ranked-clauses gpt-5.2
done
