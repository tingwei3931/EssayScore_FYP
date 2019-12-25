"""
Stores the constants that is used during training. 
Change the datapath accordingly.
"""
# The minimum and maximum scored attained for each prompt
# Used in the normalisation of the score.
# First element in the list (-1) is used for padding
MIN_SCORES = [-1, 2, 1, 0, 0, 0, 0, 0, 0, 0]
MAX_SCORES = [-1, 12, 6, 3, 3, 4, 4, 30, 60]

# Constants 
DATASET_PATH = 'datasets/' # path to the dataset
GLOVE_6B_PATH = 'embeddings/glove.6B.300d.txt' # path to the pretrained GloVe embedding with 6 billion tokens
GLOVE_42B_PATH = 'embeddings/glove.42B.300d.txt' # path to the pretrained GloVe embedding with 42 billion tokens
TRAINED_WORD2VEC_PATH = 'embeddings/'  # path to the trained word2vec embedding
PRETRAINED_WORD2VEC_PATH = 'embeddings/GoogleNews-vectors-negative300.bin' # path to the pretrained word2vec embedding
GLOVE2WORD2VEC_PATH = 'embeddings/glove2word2vec.txt' # path to converted glove2word2vec embedding
SAVED_MODEL_PATH = 'models/' # path to saved models
SAVED_FIGURE_PATH = 'figures/' # path to saved plot figures
