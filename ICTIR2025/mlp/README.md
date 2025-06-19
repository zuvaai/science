
# Binary clause classification with embeddings and MLP

This directory contains scripts to train a binary sentence or clause classifier MLP on the clause embeddings.


For each of the 10 clause types there are 20 original clauses, each with a negated and rephrased variant.


To create training examples for the MLP we take original clause and negated clause pairs for negative examples, and rephrased for positive examples. This means that for each clause type there are 20 negative and 20 positive examples.

`train_individual.py` trains an MLP per clause type using a certain number of clauses for training, for each embedding type.

The results (accuracies) are contained in the `train_individual[5,10,15].csv` files for different train/test splits.

For example, `train_individual5.csv` uses 5 clauses for training and the remaining 15 for testing. The 5 means that it uses 5/20 of the original clauses, which are paired with both negated and rephrased variants so there are actually 10 training examples. The same experiment is repeated for 10/10 (50/50% split) and 15/10 (75/25% split)

In contrast `train_one.py` trains a single MLP using the examples from all the clauses, per embedding type.


The results are contained in the `train_one_individual[5,10,15].csv` files where the evaluation is done exactly the same as above.

There are also aggregated scores (over all clause types) in the `train_one_combined[5,10,15].csv` files.

