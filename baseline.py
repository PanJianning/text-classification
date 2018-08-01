import numpy as np
np.random.seed(2018)
import pandas as pd
import xgboost as xgb
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from scipy.sparse import hstack
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error

import jieba
import re

from sklearn.externals import joblib
import os
from sklearn.model_selection import KFold
from sklearn.naive_bayes import MultinomialNB


reg_chinese = re.compile(r'[^\u4e00-\u9fa5]+')

def read_csv(filename, dir='./input/'):
	path = os.path.join(dir, filename+'.csv')
	return pd.read_csv(path)

def clean(text):
    text = re.sub(reg_chinese, ' ', text)
    return text

def chinese_tokenizer(text):
    tokens = list(jieba.cut(text))
    return tokens

word_vectorizer = TfidfVectorizer(analyzer='word',min_df=0.0,max_df=1.0,max_features=20000,
                            tokenizer=chinese_tokenizer,ngram_range=(1,2),stop_words=[' '])
char_vectorizer = TfidfVectorizer(analyzer='char',min_df=0.0,max_df=1.0,max_features=20000,
                            ngram_range=(1,2))

train_first = read_csv('train_first')
train_second = read_csv('train_second')

data = pd.concat([train_first,train_second])
all_text = data.Discuss.apply(lambda x: clean(x))

print('fitting word vectorizer...')
features_word = word_vectorizer.fit_transform(all_text)

print('fitting char vectorizer...')
features_char = char_vectorizer.fit_transform(all_text)

print('vectorize completed.')

features = hstack([features_word,features_char], format='csr')

labels = data['Score'].values-1

x_train, x_valid, y_train, y_valid = train_test_split(features,labels,
	test_size=0.4,shuffle=True,random_state=2018)

kf = KFold(n_splits=5,random_state=2018,shuffle=True)
X_trs = []
X_tes = []
Y_trs = []
Y_tes = []
for train_index, test_index in kf.split(x_train):
    X_tr,X_te= x_train[train_index], x_train[test_index]
    Y_tr,Y_te= y_train[train_index], y_train[test_index]
    X_trs.append(X_tr); X_tes.append(X_te)
    Y_trs.append(Y_tr); Y_tes.append(Y_te)

ps_lr = []
ps_nb = []
for i in range(5):
    X_tr = X_trs[i]; Y_tr = Y_trs[i]
    X_te = X_tes[i]; Y_te = Y_tes[i]
    lr = LogisticRegression(solver='sag', multi_class='multinomial')
    lr.fit(X_tr,Y_tr)
    p_lr = lr.predict_proba(X_te)
    ps_lr.append(p_lr)
    
    nb = MultinomialNB(alpha=0.01)
    nb.fit(X_tr,Y_tr)
    p_nb = nb.predict_proba(X_te)
    ps_nb.append(p_nb)
    
    print('{} fold model fitted.'.format(i))

models = []
for i in range(5):
    p1=np.concatenate(ps_lr[:i]+ps_lr[i+1:])
    p2=np.concatenate(ps_nb[:i]+ps_nb[i+1:])
    Y_train = np.concatenate(Y_tes[:i]+Y_tes[i+1:])
    X_train = np.concatenate([p1,p2],axis=1)
    X_test = np.concatenate([ps_lr[i],ps_nb[i]],axis=1)
    Y_test = Y_tes[i]

    params = {
        'eta':0.1,
        'objective': 'reg:linear',
        'lambda':2,
    }

    xgbtrain = xgb.DMatrix(X_train,Y_train+1)
    xgbtest = xgb.DMatrix(X_test,Y_test+1)

    watchlist = [(xgbtrain,'train'),(xgbtest,'test')]
    model = xgb.train(params,xgbtrain,300, watchlist,early_stopping_rounds=5)
    models.append(model)

lr = LogisticRegression(solver='sag',multi_class='multinomial')
lr.fit(x_train,y_train)
p_lr = lr.predict_proba(x_valid)

nb = MultinomialNB(alpha=0.01)
nb.fit(x_train,y_train)
p_nb = nb.predict_proba(x_valid)
print('model fitted.')

feat = np.concatenate([p_lr,p_nb],axis=1)

ys = []
for model in models:
    y = model.predict(xgb.DMatrix(feat))
    ys.append(y)

prediction = np.mean(ys,axis=0)
print(mean_squared_error(y_valid+1,prediction))
