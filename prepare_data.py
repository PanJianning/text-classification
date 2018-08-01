import numpy as np
np.random.seed(2018)
import pandas as pd
import jieba
import os
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from scipy.sparse import hstack
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.externals import joblib

import keras as ks
import keras.layers as kl
from keras.preprocessing import text, sequence

def char_tokenizer(text):
	return ' '.join(list(text))

def score(y_true, y_pred):
	rmse = np.sqrt(np.mean(y_true-y_pred)**2)
	return 1/(1+rmse)

def read_csv(filename, dir='./input/'):
	path = os.path.join(dir, filename+'.csv')
	return pd.read_csv(path)

def chinese_tokenizer(text):
	tokens = list(jieba.cut(text))
	return ' '.join(tokens)

max_features = 20000
maxlen = 150
embed_size = 128

train_first = read_csv('train_first')
train_second = read_csv('train_second')

data = pd.concat([train_first,train_second])
all_text = data.Discuss.apply(chinese_tokenizer)

tokenizer = text.Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(all_text)

features = tokenizer.texts_to_sequences(all_text)
features = sequence.pad_sequences(features,maxlen=maxlen)
labels = data.Score.values 

joblib.dump(features, './tempdata/features.npy')
joblib.dump(labels, './tempdata/labels.npy')
joblib.dump(tokenizer.word_index, './tempdata/word_index.npy')
