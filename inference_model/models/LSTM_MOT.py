from keras.layers import Embedding, LSTM, Dense, Dropout, Conv1D, Lambda
from keras.models import Sequential
from constants import *
from preprocess import nltk_tokenizer, buildEmbeddingMatrix
from keras import backend as K 
import keras as k

class LSTM_MOT:
    """
        Implements LSTM Network with Mean Over Time layer.
        Adapted from A Neural Approach to Automated Essay Scoring. (Taghipour and Ng, 2016)
    """
    # Define hyperparameters for the network
    _dropout_rate = 0.4
    _activation_func = 'sigmoid'
    _loss = 'mean_squared_error'
    _optimizer = 'adam'
    _metrics = ["mae", "mse"]

    def getModel(self, embedding_dimen=300, essay_length=500, embedding_model=GLOVE_6B_PATH):
        """ Creates and returns the neural network model.
            Parameters:
                embedding_dimen: the dimension of the word embedding. Will be in the shape of (num_essays, 300).
                essay_length:    the length of essay fed into the network. Extras will be trimmed and padding will be                      added otherwise.
                embedding_model: the word embedding used to create the model. Default will be the 6 billion tokens GloVe                   model.
            Returns:
                A model with the following architecture:
                [Embedding] -> [Convolution 1D] -> [LSTM] -> [MOT] -> [Sigmoid] -> [Prediction Score]
        """
        # Retrieves vocab summary from nltk tokenzier
        vocab_size = len(nltk_tokenizer.word_index) + 1
        embedding_matrix = buildEmbeddingMatrix(embedding_model, nltk_tokenizer)

        # Start stacking the layers 
        model = Sequential()
        # Freeze the embedding layer so that the weights do not get adjusted during training process 
        model.add(Embedding(vocab_size, embedding_dimen, weights=[embedding_matrix], input_length=essay_length, 
                            trainable=False, mask_zero=False))
        model.add(Conv1D(filters=50, kernel_size=5, padding='same'))
        model.add(LSTM(300, dropout=0.4, recurrent_dropout=0.4, return_sequences=True))
        # Add a mean over time layer
        model.add(Lambda(lambda x: K.mean(x, axis=1)))
        model.add(Dropout(0.4))
        model.add(Dense(1, activation=self._activation_func, activity_regularizer=k.regularizers.l2(0.0)))
        model.compile(loss=self._loss, optimizer=self._optimizer, metrics=self._metrics)

        print("--- Model Summary ---")
        print(model.summary())
        return model
    