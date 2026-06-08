#!/bin/bash

rm tmp
echo "Input,Output" > tmp
for i in parsed-obligations/usagegpt-4.1*
do
    cat  $i  >> tmp
done
echo "Parsed (4.1)" $(python3 calc_usage_values.py tmp)

rm tmp
echo "Input,Output" > tmp
for i in parsed-obligations/usagegpt-5.2*
do
    cat $i  >> tmp
done
echo "Parsed (5.2)" $(python3 calc_usage_values.py tmp)

# For One and Two prompt methods we ignore the first entry since it correponds template-template comparison
rm tmp
echo "Input,Output" > tmp
for i in aligned-clauses-two-stage/usage-matched-gpt-4.1-gpt-5.2-*
do
    gtail -n +2 $i >> tmp
done
echo "2P (4.1|5.2)" $(python3 calc_usage_values.py tmp)

rm tmp
echo "Input,Output" > tmp
for i in aligned-clauses-two-stage/usage-matched-gpt-5.2-gpt-4.1-*
do
    gtail -n +2 $i >> tmp
done
echo "2P (5.2|4.1)" $(python3 calc_usage_values.py tmp)

rm tmp
echo "Input,Output" > tmp
for i in aligned-clauses-two-stage/usage-matched-gpt-5.2-gpt-5.2-*
do
    gtail -n +2 $i >> tmp
done
echo "2P (5.2)" $(python3 calc_usage_values.py tmp)

rm tmp
echo "Input,Output" > tmp
for i in aligned-clauses-one-stage/usage-matched-gpt-5.2-*
do
    gtail -n +2 $i >> tmp
done
echo "1P (5.2)" $(python3 calc_usage_values.py tmp)

# Listwise has only single values since all prompts are compared simultaneously
rm tmp
echo "Input,Output" > tmp
for i in ranked-clauses/usage-ranked-gpt-5.2-*
do
    cat $i  >> tmp
done
echo "L (5.2)" $(python3 calc_usage_values.py tmp)

# All clauses have to be embedded and are counted
rm tmp
echo "Value" > tmp
for i in openai/text-embedding-3-large/*; do
    cat $i/usage.csv | awk '{print $1}' >> tmp; 
done
echo "O" $(python3 calc_agg_values.py tmp)