import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd

import os
import sys

# get num_train from command line argument
num_train = int(sys.argv[1])
   
def load_embeddings(task_path):
    orig_embs = np.asarray(np.load(f"{task_path}/Original.pkl", allow_pickle=True))
    rephrased_embs = np.asarray(np.load(f"{task_path}/DifferentMeaningMinimalChanges.pkl", allow_pickle=True))
    negated_embs = np.asarray(np.load(f"{task_path}/SameMeaningWordedDifferently.pkl", allow_pickle=True))

    # positive train examples
    num_examples = len(orig_embs)

    embedding_dim = orig_embs.shape[1] 

    x_orig_train_pos = orig_embs[:num_train]
    x_reph_train_pos = rephrased_embs[:num_train]
    y_train_pos = np.ones((num_train, 1))

    # negative train examples
    x_orig_train_neg = orig_embs[:num_train]
    x_ngt_train_neg = negated_embs[:num_train]
    y_train_neg = np.zeros((num_train, 1))

    # Combine positive and negative examples for training
    x1_train = np.concatenate((x_orig_train_pos, x_orig_train_neg), axis=0)
    x2_train = np.concatenate((x_reph_train_pos, x_ngt_train_neg), axis=0)
    y_train = np.concatenate((y_train_pos, y_train_neg), axis=0)

    # shuffle the training data with fixed random seed
    np.random.seed(42)
    indices = np.arange(len(y_train))
    np.random.shuffle(indices)
    x1_train = x1_train[indices]
    x2_train = x2_train[indices]
    y_train = y_train[indices]

    # positive test examples
    x_orig_test_pos = orig_embs[num_train:]
    x_reph_test_pos = rephrased_embs[num_train:]
    y_test_pos = np.ones((num_examples-num_train, 1))
    # negative test examples
    x_orig_test_neg = orig_embs[num_train:]
    x_reph_test_neg = negated_embs[num_train:]
    y_test_neg = np.zeros((num_examples-num_train, 1))

    x1_test = np.concatenate((x_orig_test_pos, x_orig_test_neg), axis=0)
    x2_test = np.concatenate((x_reph_test_pos, x_reph_test_neg), axis=0)
    y_test = np.concatenate((y_test_pos, y_test_neg), axis=0)

    # shuffle the test data with fixed random seed
    np.random.seed(42)
    indices_test = np.arange(len(y_test))
    np.random.shuffle(indices_test)
    x1_test = x1_test[indices_test]
    x2_test = x2_test[indices_test]
    y_test = y_test[indices_test]

    return x1_train, x2_train, y_train, x1_test, x2_test, y_test, embedding_dim

def train(task_path):

    X1_train, X2_train, y_train, X1_val, X2_val, y_val, embedding_dim = load_embeddings(task_path)

    # Define the model
    input_1 = keras.Input(shape=(embedding_dim,), name='sentence1_embedding')
    input_2 = keras.Input(shape=(embedding_dim,), name='sentence2_embedding')

    # Concatenate the two embeddings
    x = layers.Concatenate()([input_1, input_2])

    # MLP layers
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation='relu')(x)

    # Output layer
    output = layers.Dense(1, activation='sigmoid')(x)

    # Build the model
    model = keras.Model(inputs=[input_1, input_2], outputs=output)

    # Compile the model
    model.compile(optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy'])

    # Print model summary
    model.summary()

    # Train the model
    history = model.fit(
        [X1_train, X2_train], y_train,
        validation_data=([X1_val, X2_val], y_val),
        epochs=100,
        batch_size=10
    )

    # print the final validation accuracy
    val_loss, val_accuracy = model.evaluate([X1_val, X2_val], y_val)

    # get the sigmoid output for the validation set
    y_val_pred = model.predict([X1_val, X2_val])
    # write the predictions to a csv file
    pred_df = pd.DataFrame({
        'y_true': y_val.flatten(),
        'y_pred': y_val_pred.flatten()
    })
    # remove ../ from the task path
    task_path = task_path.replace("../", "")
    # make directory for predictions
    os.makedirs(f"predictions_individual/{task_path}", exist_ok=True)
    pred_df.to_csv(f"predictions_individual/{task_path}/predictions_{num_train}.csv", index=False)
    return val_loss, val_accuracy

if __name__ == "__main__":



    # embedding models
    experiment_paths = [
        "models/fasttext/crawl-300d-2M-subword-norm",

        "models/nvidia/nv-embed-v2-norm",

        "models/openai/text-embedding-3-large",
        
        "models/gemini/text-embedding-004",

        "models/legal-bert/legal-bert-base-uncased-meanpool-norm",

    ]

    # clause types
    clause_types = [
        "Assignment",
        "Transfer of Data",
        "Exclusivity",
        "Non-Solicit",
        "Permitted Use of Data",
       "Audit Right",
        "License Grant",
        "MFN",
        "Publicity",
        "Termination for Convenience"
    ]

    scores = []

    for e in experiment_paths:
        for c in clause_types:
            task_path = f"../embeddings/{e}/{c}"
            _, val_accuracy = train(task_path)
            # create dict for each experiment/clause type and store the validation accuracy
            scores.append({
                'experiment': e.split('/')[-1],
                'clause_type': c,
                'val_accuracy': val_accuracy
            })

    # write the scores to csv
    df = pd.DataFrame(scores)
    # round the float values to 4 decimal places
    df['val_accuracy'] = df['val_accuracy'].round(4)
    df.to_csv(f"train_individual{num_train}.csv", index=False)

