from constants import GLOVE_6B_PATH
from keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from keras.models import Sequential
import keras.regularizers
from preprocess import nltk_tokenizer, buildEmbeddingMatrix

class BiLSTM:
    """
        Implements a Bidirectional LSTM Network.
        Adapted from A Neural Approach to Automated Essay Scoring. (Taghipour and Ng, 2016)
    """
    # Define hyperparameters for the network
    _dropout_rate = 0.4
    _activation_func = 'sigmoid'
    _loss = 'mean_squared_error'
    _optimizer = 'adam'
    _metrics = ["mae", "mse"]

    def getModel(self, embedding_dimen=300, essay_len=500, embedding_model=GLOVE_6B_PATH):
        """ Returns compiled model."""
        vocabulary_size = len(nltk_tokenizer.word_index) + 1
        embedding_matrix = buildEmbeddingMatrix(embedding_model, nltk_tokenizer)

        model = Sequential()
        model.add(Embedding(vocabulary_size, embedding_dimen, weights=[embedding_matrix], input_length=essay_len, trainable=False, mask_zero=True))
        model.add(Bidirectional(LSTM(150, dropout=0.4, recurrent_dropout=0.4)))
        model.add(Dropout(0.4))
        model.add(Dense(1, activation=self._activation_func, activity_regularizer=keras.regularizers.l2(0.01)))
        model.compile(loss=self._loss, optimizer=self._optimizer, metrics=self._metrics)
        print("--- Model Summary ---")
        print(model.summary())
        return model