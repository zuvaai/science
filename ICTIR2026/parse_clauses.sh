#!/bin/bash


mkdir -p parsed-obligations
for i in clauses/*;
do
  python3 obligation_parsing.py ${i} parsed-obligations gpt-4.1
  python3 obligation_parsing.py ${i} parsed-obligations gpt-5.2
done
