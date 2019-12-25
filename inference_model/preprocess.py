import os 
from keras.backend.tensorflow_backend import set_session
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from gensim.models import word2vec
import matplotlib.pyplot as plt
import nltk.data
import re
from nltk.corpus import stopwords
from sklearn.metrics import cohen_kappa_score
import logging
from constants import GLOVE_42B_PATH, GLOVE_6B_PATH, DATASET_PATH, MAX_SCORES, MIN_SCORES 
from time import time
from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
import gensim
import pprint
import pickle

# Setup logging to monitor progress 
logging.basicConfig(format="%(levelname)s - %(asctime)s: %(message)s", datefmt= '%H:%M:%S', level=logging.INFO)

nltk_tokenizer = Tokenizer()

def readData(path_to_dataset, train_size=0.8, validation_size=0.2):
    """ Reads data from csv file and perform train, test and validation split.
        Parameters:
            path_to_dataset: absolute path to the folder containing dataset.
            train_size:      training dataset size. Default is 80% of total dataset.
            validation_size: validation dataset size. Defailt is 20% of the total dataset.
        Returns:
            A tuple containing training, testing and validation dataset.    
    """
    data = pd.read_csv(os.path.join(path_to_dataset, 'training_set_rel3.tsv'), sep='\t', encoding='ISO-8859-1')
    # Drop columns that has null value 
    data = data.dropna(axis=1)
    # Only take 4 columns of data from the dataset: essay_id, essay_set, essay, domain1_score
    data = data[['essay_id', 'essay_set', 'essay', 'domain1_score']]
    # Perform 80:20 train-test split on the training data
    train_set, test_set = train_test_split(data, train_size=train_size, random_state=0)
    # Split the 80% training set further into 60:20
    training_set, validation_set = train_test_split(train_set, test_size=validation_size, random_state=0)
    return training_set, test_set, validation_set 

def normaliseScores(essays):
    """ Accepts the dataset as input and normalises the scores in the range [0-1] using
        min-max normalisation.
        Parameters:
            essays: dataset containing essay samples together with score.
        Returns:
            A numpy array containing the normalised scores in the range of [0-1].
    """
    normalised_scores = []
    for index, essay in essays.iterrows():
        score = essay['domain1_score']
        # essay_set refers to the prompt(topic) of the essay
        essay_set = essay['essay_set']
        # Perform min-max normalization on the scores to get range in [0-1]
        normalised_score = (score - MIN_SCORES[essay_set]) / (MAX_SCORES[essay_set] - MIN_SCORES[essay_set])
        normalised_scores.append(normalised_score)
    return np.array(normalised_scores)


def tokenize(essays, essay_len=500):
    """ Tokenize the essays into their own vector representations.
        Parameters:
            essays:    dataset containing essay samples together with score.
            essay_len: length of the essay that will be tokenized. Default is 500 words.
        Returns:
            A tuple containing data and label tensors which are the feature vectors ready to be trained.
    """
    # normalise scores into range of [0-1]
    normalised_scores = normaliseScores(essays)
    logging.info("Begin tokenization of essays...")
    # Begin the timer 
    t = time()
    # Fit_on_texts method creates the vocabulary index based on word frequency
    nltk_tokenizer.fit_on_texts(essays['essay'])
    logging.info("Tokenization Done!")
    logging.info("Time taken to tokenize essays: {} seconds ".format(round(time() - t), 2))
    logging.info("Tokenization Summary: ")
    logging.info("Number of unique words found: {}".format(len(nltk_tokenizer.word_index)))
    logging.info("Number of essays used for tokenisation: {}".format(nltk_tokenizer.document_count))    
    # Transform each essay into a sequence of integers by converting each word to their unique integer asasigned in word_index
    sequence_vectors = nltk_tokenizer.texts_to_sequences(essays['essay'])
    print(sequence_vectors[0])
    # Pad the vectors so that all of them is essay_len long
    data = pad_sequences(sequence_vectors, maxlen=essay_len)
    label = normalised_scores
    # Reshape the label tensor to (num_essays, 1)
    label = np.reshape(label, (len(label), 1))
    logging.info('Shape of data tensor: {}'.format(data.shape))
    logging.info('Shape of label tensor: {}'.format(label.shape))
    return data, label

def buildEmbeddingMatrix(path_to_gloVe, tokenizer, embedding_dimen=300):
    """ Reads in the pretrained gloVe vectors and builds the embedding layer.
        Parameters:
            path_to_glove:   Absolute path to the folder containing GloVe word embedding.
            tokenizer:       NLTK Tokenizer object containing the word to integer dictionary. 
            embedding_dimen: Dimension of the GloVe embedding. Each word is represented with 300 dimensions. 
        Returns:
            embedding_matrix: A GloVe embedding matrix containing the representations of words found in the corpora.
    """
    logging.info("Loading GloVe vector model..")
    t = time()
    # Loads the gloVe model into a dictionary
    with open(path_to_gloVe, encoding='utf8') as file:
        embeddings = dict()
        for line in file:
            values = line.split()
            # key is the word, value is the numpy array for the corresponding word vector
            embeddings[values[0]] = np.asarray(values[1:], 'float32')
    # Create a 2D tensor of shape(num_unique_words+1, embedding_dimen) (Index 0 is used for padding)
    embedding_matrix = np.zeros((len(nltk_tokenizer.word_index) + 1, embedding_dimen))
    word_found_in_embedding = 0
    for word, index in nltk_tokenizer.word_index.items():
        embedding_vector = embeddings.get(word)
        # Only populate word vectors that exist in GloVe model,
        # words not found (e.g: spelling error) will be padded with zeroes as their word vector
        if embedding_vector is not None:
            embedding_matrix[index] = embedding_vector
            word_found_in_embedding += 1
    logging.info("Done!")
    logging.info("Loaded {} word vectors into the embedding.".format(len(embedding_matrix)))
    logging.info("Found {} word vectors that exist in the GloVe model.".format(word_found_in_embedding))
    logging.info("Time taken to load pre-trained GloVe model: {} mins".format(round(((time() - t) / 60), 2)))
    return embedding_matrix

if __name__ == '__main__':
    # For testing out the preprocess methods
    nltk_tokenizer = Tokenizer()
    train_set, test_set, validation_set = readData(DATASET_PATH)
    X_train, Y_train = tokenize(train_set)
    X_validation, Y_validation = tokenize(validation_set)
    X_test, Y_test = tokenize(test_set)
    # Export out the tokenizer using pickle
    with open('tokenizer.pickle', 'wb') as handle:
        pickle.dump(nltk_tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    embeddingMatrix = buildEmbeddingMatrix(GLOVE_6B_PATH, nltk_tokenizer)
    print(embeddingMatrix.shape)
    


