import numpy as np
np.random.seed(2018)
import keras as ks
import keras.layers as kl
from keras import regularizers
import gensim

max_features = 20000
maxlen = 150
embed_size = 128

word_index = joblib.load("./tempdata/word_index.npy")
wv = gensim.models.KeyedVectors.load_word2vec_format('./tempdata/word2vec.bin',binary=True)

vocab_size = len(wv.vocab)
embedding_matrix = np.zeros((max_features, embed_size))
for word, i in word_index.items():
    if i >= max_features or word not in wv.vocab: continue
    embedding_vector = wv.word_vec(word)
    if embedding_vector is not None: embedding_matrix[i] = embedding_vector

def get_model():
  pass
