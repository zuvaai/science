import numpy as np
import tensorflow as t
from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd

import sys
import os

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

def train(X1_train, X2_train, y_train, X1_val, X2_val, y_val, embedding_dim):

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
    val_loss, val_accuracy = model.evaluate([X1_val, X2_val], y_val)
    return val_loss, val_accuracy, model

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

    
    individual_scores = []
    combined_scores = []

    for e in experiment_paths:

        individual_test_data = {}
        # to combine train and test data for all clause types
        all_x1_train, all_x2_train, all_y_train = [], [], []
        all_x1_test, all_x2_test, all_y_test = [], [], []
        for c in clause_types:
            task_path = f"../embeddings/{e}/{c}"
            x1_train, x2_train, y_train, x1_test, x2_test, y_test, _ = load_embeddings(task_path)

            all_x1_train.append(x1_train)
            all_x2_train.append(x2_train)
            all_y_train.append(y_train)

            all_x1_test.append(x1_test)
            all_x2_test.append(x2_test)
            all_y_test.append(y_test)

            # store test sets to get scores per clause type
            individual_test_data[c] = {
                'x1_test': x1_test,
                'x2_test': x2_test,
                'y_test': y_test,
            }

        # Concatenate all clause types for training and testing
        x1_train = np.concatenate(all_x1_train, axis=0)
        x2_train = np.concatenate(all_x2_train, axis=0)
        y_train = np.concatenate(all_y_train, axis=0)
        x1_test = np.concatenate(all_x1_test, axis=0)
        x2_test = np.concatenate(all_x2_test, axis=0)
        y_test = np.concatenate(all_y_test, axis=0)

        # Get the embedding dimension from the first clause type
        embedding_dim = all_x1_train[0].shape[1]

        # shuffle the training data with fixed random seed
        np.random.seed(42)
        indices = np.arange(len(y_train))
        np.random.shuffle(indices)
        x1_train = x1_train[indices]
        x2_train = x2_train[indices]
        y_train = y_train[indices]

        # shuffle the test data with fixed random seed
        np.random.seed(42)
        indices_test = np.arange(len(y_test))
        np.random.shuffle(indices_test)
        x1_test = x1_test[indices_test]
        x2_test = x2_test[indices_test]
        y_test = y_test[indices_test]

        # Train the model
        val_loss, val_accuracy, model = train(x1_train, x2_train, y_train, x1_test, x2_test, y_test, embedding_dim)

        # Store the scores for the combined model
        combined_scores.append({
            'experiment': e.split('/')[-1],
            'clause_type': "all",
            'val_accuracy': val_accuracy
        })

        # get scores for individual clause types
        for c in clause_types:
            test_data = individual_test_data[c]
            x1_test = test_data['x1_test']
            x2_test = test_data['x2_test']
            y_test = test_data['y_test']

            # Evaluate the model on the individual clause type
            _, val_accuracy = model.evaluate([x1_test, x2_test], y_test)

            # Store the scores for the individual clause type
            individual_scores.append( {
                'experiment': e.split('/')[-1],
                'clause_type': c,
                'val_accuracy': val_accuracy
            })

                # get the sigmoid output for the validation set
            y_test_pred = model.predict([x1_test, x2_test])
            # write the predictions to a csv file
            pred_df = pd.DataFrame({
                'y_true': y_test.flatten(),
                'y_pred': y_test_pred.flatten()
            })
            # make directory for predictions
            task_path = task_path.replace("../", "")
            os.makedirs(f"predictions_one/{task_path}", exist_ok=True)
            pred_df.to_csv(f"predictions_one/{task_path}/predictions_{num_train}.csv", index=False) 
    

    df = pd.DataFrame(combined_scores)
    df['val_accuracy'] = df['val_accuracy'].round(4)
    df.to_csv(f"train_one_combined{num_train}.csv", index=False)


    df = pd.DataFrame(individual_scores)
    df['val_accuracy'] = df['val_accuracy'].round(4)
    df.to_csv(f"train_one_indivdual{num_train}.csv", index=False)
