import os 
from keras.backend.tensorflow_backend import set_session
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow.compat.v1 as tf
from gensim.models import word2vec
import matplotlib.pyplot as plt
import nltk.data
import re
from nltk.corpus import stopwords
from sklearn.metrics import cohen_kappa_score
import logging
from constants import * 
from visualize import * 
from time import time
from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
import gensim

# Disable tf2.0 behavior (We are only using tf1.0 features for this project)
tf.disable_v2_behavior()

def convertGloVe(path_to_glove, path_to_output):
    """
    Converts gloVe model to word2vec embedding model and returns the converted word2vec model.
    Link:
    https://radimrehurek.com/gensim/scripts/glove2word2vec.html
    """
    # Begin conversion process
    glove2word2vec(glove_input_file=path_to_glove, word2vec_output_file=path_to_output)

    model = KeyedVectors.load_word2vec_format(path_to_output, binary=False)
    return model

def load_word2vec(path_to_word2vec):
    model = KeyedVectors.load_word2vec_format(path_to_word2vec, binary=True)
    return model

def essay_to_wordlist(essay, remove_stopwords=True):
    """
    Remove the anonymisation labels (@DATE etc) and tokenise
    essay into words.
    """
    # replace all non alphabetic characters with whitespace
    essay = re.sub("[^a-zA-Z]", " ", essay)
    #split essay into words
    words = essay.lower().split()
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        # exclude stopwords in the final list
        words = [w for w in words if not w in stops]
    return (words)

def essay_to_sentences(essay, remove_stopwords=True):
    """
    Tokenise the essay into sentences and further tokenize the sentences into words. 
    """
    # Punkt is a sentence boundary detection algorithm
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

    # Tokenise the essay into raw sentences
    raw_sentences = tokenizer.tokenize(essay.strip())
    words_in_sentences = []
    for raw_sentence in raw_sentences:
        if len(raw_sentence) > 0:
            words_in_sentences.append(essay_to_wordlist(raw_sentence, remove_stopwords))
    return words_in_sentences

def parse_essays(essays):
    """
    Begin parsing each essays into word tokens.
    """
    print("Begin parsing the essays into words...")
    # Begin the timer
    t = time()
    tokens = []
    for essay in essays:
        tokens += essay_to_sentences(essay, remove_stopwords=True)
    print("Complete!")
    print("Time taken to parse all the essays: {} mins ".format(round((time() - t) / 60), 2))
    return tokens

def train_word2vec(tokens):
    # initialise variables for word2vec model
    num_features = 300 # dimensions for the trained vectors
    min_word_count = 40 # Specify that the word must appear at least 40 times in the corpus to be included in training
    num_workers = 4 # num of worker threads, parallelisation to speed up training 
    context = 10 # maximum distance between the current and predicted word within a sentence.
    downsampling = 1e-3 # the threshold for configuring which higher-frequency words are randomly downsampled

    print("Begin training word2vec model...")
    # Begin the timer 
    t = time()
    # Here, the word2vec model uses the skip-gram approach to train the embedding matrix. other alternatives include CBOW 
    model = word2vec.Word2Vec(tokens, workers=num_workers, size=num_features, min_count=min_word_count, window=context, sample=downsampling, iter=50)
    model.init_sims(replace=True)
    # Export the trained word2vec model
    model_name = "trained_word2vec"
    model.save(os.path.join(TRAINED_WORD2VEC_PATH, model_name))
    print("Word2vec model training completed!")
    print("Time taken to train word2vec model: {} mins ".format(round((time() - t) / 60), 2))
    return model

def buildFeatureVec(words, model, num_features):
    """ Build feature vector from bag of words (BOW) model and the trained Word2Vec model"""
    # Initialise numpy array of zeros
    feature_vec = np.zeros((num_features,), dtype="float32")
    num_words = 0
    index2word = set(model.wv.index2word)
    for word in words:
        if word in index2word:
            num_words = num_words + 1
            feature_vec = np.add(feature_vec, model[word])
    feature_vec = np.divide(feature_vec, num_words)
    return feature_vec

def getAvgFeatureVecs(essays, model, num_features=300):
    """
    Generates the averaged feature vector for each essay to be used in the embedding layer.
    """
    counter = 0
    essayFeatureVecs = np.zeros((len(essays), num_features), dtype="float32")
    for essay in essays:
        essayFeatureVecs[counter] = buildFeatureVec(essay, model, num_features)
        counter = counter + 1
    return essayFeatureVecs

''' # Create avg. feature vector for each training and testing essays using word2vec model
cleaned_training_essays = []
for essay in X_train:
    cleaned_training_essays.append(essay_to_wordlist(essay, remove_stopwords=True))
#print(cleaned_training_essays[0])
trainingFeatureVec = getAvgFeatureVecs(cleaned_training_essays, model)
print("Training essay vector shape:", trainingFeatureVec.shape)
print(trainingFeatureVec[0])

cleaned_testing_essays = []
for essay in X_test:
    cleaned_testing_essays.append(essay_to_wordlist(essay, remove_stopwords=True))
print(cleaned_testing_essays[0])
testingFeatureVec = getAvgFeatureVecs(cleaned_testing_essays, model)
print("Testing essay vector shape:", testingFeatureVec.shape)
'''