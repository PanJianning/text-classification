import numpy as np

np.random.seed(2018)

import jieba
import pandas as pd
import os
import re
import gensim

# remove non-chinese character
def clean(text):
	reg_chinese = re.compile(r'[^\u4e00-\u9fa5]+')
	text = re.sub(reg_chinese, ' ', text)
	return text

def read_csv(filename, dir='./input/'):
	path = os.path.join(dir, filename+'.csv')
	return pd.read_csv(path)

def chinese_tokenizer(text):
	return list(jieba.cut(text))

max_features = 20000
maxlen = 150
embed_size = 128

train_first = read_csv('train_first')
train_second = read_csv('train_second')

data = pd.concat([train_first,train_second])
all_text = data.Discuss.apply(clean)
all_text = all_text.drop_duplicates()

tokenized_sentences = []
for text in all_text:
	tokenized_sentences.append(chinese_tokenizer(text))

model = gensim.models.Word2Vec(tokenized_sentences,size=embed_size,window=4,min_count=1,negative=3,
	sg=1,sample=0.001,hs=0,workers=4,iter=15)
model.wv.save_word2vec_format("./tempdata/word2vec.bin",binary=True)
