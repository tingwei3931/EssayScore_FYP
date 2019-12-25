from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import savefig
import os
from constants import SAVED_FIGURE_PATH
from preprocess import *

def visualize_similar(model, word, nearest_n, is_glove=False):
    """
    Visualise the trained word2vec model. This function will plot the nearest n words to the word provided.
    """
    # Find the top-N most similar words by computing the cosine similarity between words
    nearest_n_words = model.most_similar(word, topn=nearest_n)
    words_to_plot = [word]
    # Create a tensor of size nearest_n + 1 (including the specified word)
    words_vec = np.zeros((nearest_n+1, 300))
    words_vec[0, :] = model[word]

    # Populate the nparray with trained word2vec vectors
    for i in range(0, len(nearest_n_words)):
        words_to_plot.append(nearest_n_words[i][0])
        words_vec[i+1, :] = model[nearest_n_words[i][0]]
    
    # Perform dimensionality reduction from 300 to 2 for plotting using principal component analysis (PCA)
    pca =  PCA(n_components=2)
    pca.fit(words_vec)
    X = pca.transform(words_vec)
    x_axis = X[:, 0]
    y_axis = X[:, 1]

    # Specify the plot to be 8 by 6 inches
    plt.figure(figsize=(8,6))
    plt.scatter(x_axis, y_axis, marker='^')
    plt.xlabel('PCA1')
    plt.ylabel('PCA2')
    title = 'glove2word2vec ' if is_glove else 'word2vec '
    plt.title('Nearest '+ title + 'embedding for ' + word)
    for index, w in enumerate(words_to_plot):
        plt.annotate(w, xy=(x_axis[index], y_axis[index]), xytext=(3, 3), textcoords='offset points', 
        ha='left', va='top')
    # Export the plot as png 
    savefig(os.path.join(SAVED_FIGURE_PATH, word + "_" + title + "_embedding.png"), bbox_inches='tight')
    plt.show()


    