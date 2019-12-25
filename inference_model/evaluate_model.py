from constants import MIN_SCORES, MAX_SCORES, DATASET_PATH
import datetime
import json
import numpy as np
import os
import logging 
from preprocess import tokenize
from keras.models import model_from_json
from sklearn.model_selection import train_test_split
from keras import backend as K
import pandas as pd
from quadratic_weighted_kappa import quadratic_weighted_kappa as qwk
import pprint
import pickle
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

# Setup logging to monitor progress 
logging.basicConfig(format="%(levelname)s - %(asctime)s: %(message)s", datefmt= '%H:%M:%S', level=logging.INFO)

class QWKScore:
    def __init__(self, essays, model):
        self.essays = essays
        self.model = model
        np.set_printoptions(precision=3)

    def calcQWK(self):
        # For each essay set, calculate the QWK scores
        qwk_scores = []
        num_essays = []
        logging.info("Calculating QWK Scores from LSTM-MOT model.json...")

        for prompt in range(1, 9):
            # Filter essays based on the prompt
            essays_in_prompt = self.essays[self.essays['essay_set'] == prompt]
            
            with open('tokenizer.pickle', 'rb') as handle:
                nltk_tokenizer = pickle.load(handle)
                seqeuence_vectors = nltk_tokenizer.texts_to_sequences(essays_in_prompt['essay'])
                X = pad_sequences(seqeuence_vectors, maxlen=500)
            y_true = essays_in_prompt['domain1_score'].values

            # Output from the model is normalised, need to scale back to respective scale
            normalised_pred = model.predict(X)
            normalised_pred = np.array(normalised_pred)
            y_pred = np.around((normalised_pred * (MAX_SCORES[prompt] - MIN_SCORES[prompt])) + MIN_SCORES[prompt])

            # Convert y_true and y_pred to integer scalar arrays
            y_true_int = np.rint(y_true).astype('int32')
            y_pred_int = np.rint(y_pred).astype('int32').ravel() # flatten the 2D array into 1D for looping
            #pprint.pprint(y_true_int)
            #pprint.pprint(y_pred_int)
            qwk_score = qwk(y_true_int, y_pred_int, MIN_SCORES[prompt], MAX_SCORES[prompt])
            
            qwk_scores.append(qwk_score)
            num_essays.append(len(essays_in_prompt))

            logging.info("Quadratic Kappa Score for Set {}: {}".format(str(prompt), str(qwk_scores[prompt-1])))

        # Convert both of qwk_scores and num_essays list to numpy arrays
        qwk_scores = np.array(qwk_scores)
        num_essays = np.array(num_essays)

        # Calculate the mean of all QWK scores
        avg_weighted_qwk_score = np.sum(qwk_scores * num_essays) / np.sum(num_essays)

        # Print final accuracy score
        logging.info("Average Weighted QWK score: {}".format(str(avg_weighted_qwk_score)))

if __name__ == '__main__':
    with open('model_LSTMMOT.json','r') as f:
        model = model_from_json(f.read(), custom_objects={"k": K})
    # load the model for evaluation 
    model.load_weights('model_weights/LSTMMOT-weights.best.hdf5')
    # summarize loaded model 
    model.summary()
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=["mae", "mse"])
    #model.save('model_LSTMMOT.h5')
    data = pd.read_csv(os.path.join(DATASET_PATH, 'training_set_rel3.tsv'), sep='\t', encoding='ISO-8859-1')
    # Drop columns that has null value 
    data = data.dropna(axis=1)
    # Only take 4 columns of data from the dataset: essay_id, essay_set, essay, domain1_score
    data = data[['essay_id', 'essay_set', 'essay', 'domain1_score']]
    # Perform 80:20 train-test split on the training data
    train_set, test_set = train_test_split(data, train_size=0.8, random_state=0)
    qwkCalculator = QWKScore(test_set, model)
    qwkCalculator.calcQWK()
