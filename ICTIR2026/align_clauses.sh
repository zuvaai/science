#!/bin/bash


mkdir -p aligned-clauses-two-stage
for i in parsed-obligations/*;
do
  python3 obligation_matching.py ${i} aligned-clauses-two-stage gpt-4.1
  python3 obligation_matching.py ${i} aligned-clauses-two-stage gpt-5.2
done

for i in parsed-obligations/*;
do
  python3 obligation_matching.py ${i} aligned-clauses-two-stage  gpt-4.1
  python3 obligation_matching.py ${i} aligned-clauses-two-stage  gpt-5.2
done

mkdir -p aligned-clauses-one-stage
for i in clauses/*;
do
  python3 obligation_alignment.py ${i} aligned-clauses-one-stage gpt-4.1
  python3 obligation_alignment.py ${i} aligned-clauses-one-stage gpt-5.2
done
