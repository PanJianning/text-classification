# text-classification
A text classification project using the dataset from YunYi Cup

### 0. Problem
The data set contains 22w (x,y) records.

x is a comment text on some tourist attractions, y is the corresponding comment score range from 1 to 5

![data example](http://ok669z6cd.bkt.clouddn.com/data_eg.png?attname=)

This is a supervised learning problem. At test time, given a comment, we want to predict it's score.

Since the score is an ordinal variable, it turns out that regression is more suitable than classification here.
**So the metric is mse**

Note: I use 60% of the data as training set and 40% as validation set, no test set.

### 1. Baseline Model
Stacking: 
1. first layer:  Logistic regression and Naive Bayes
2. second layer: xgboost

validation mse: 0.405770

### 2. NN Model
Some common parameters:
```python
max_features = 20000
maxlen = 150
embed_size = 128
```
The embedding_matrix has 2 option:

(1) Random embedding

initialize the embedding matrix randomly, and then modified during training. 

(2) Static word2vec embedding

The word2vec embedding is trained from the 22w comment texts with gensim word2vec model.

#### 2.1 NN Baseline Model
```python
def get_model():
    inp = kl.Input(shape=(maxlen,))
    embed = kl.Embedding(max_features,embed_size,weights=[embedding_matrix],trainable=False)(inp)
    avgpool = kl.GlobalAvgPool1D()(embed)
    out = kl.Dense(256,activation='relu')(avgpool)
    out = kl.Dense(128,activation='relu')(out)
    out = kl.Dense(1)(out)
    model = ks.models.Model(inp,out)
    model.compile(loss="mse", optimizer='adam', metrics=[])
    return model
```
![nn_baseline](http://ok669z6cd.bkt.clouddn.com/nn_baseline.PNG)

The best validation mse is 0.432069
#### 3.1 CNN Model
##### 3.1.1 Vanilla TextCNN
```python
def get_model():
    inp = kl.Input(shape=(maxlen,))
    embed = kl.Embedding(max_features, embed_size, trainable=True)(inp)
    conv1 = kl.Convolution1D(filters=100, kernel_size=3, strides=1, 
        kernel_initializer='glorot_uniform', activation='relu')(embed)
    conv2 = kl.Convolution1D(filters=100, kernel_size=4, strides=1, 
        kernel_initializer='glorot_uniform', activation='relu')(embed) 
    conv3 = kl.Convolution1D(filters=100, kernel_size=5, strides=1, 
        kernel_initializer='glorot_uniform', activation='relu')(embed)
    maxpool1 = kl.GlobalMaxPool1D()(conv1)
    maxpool2 = kl.GlobalMaxPool1D()(conv2)
    maxpool3 = kl.GlobalMaxPool1D()(conv3)
    out = kl.concatenate([maxpool1,maxpool2,maxpool3])
    out = kl.Dense(1)(out)
    model = ks.models.Model(inp,out)
    model.compile(loss="mse", optimizer='adam', metrics=[])
    return model
```
![textcnn-rand](http://ok669z6cd.bkt.clouddn.com/cnn_rand.png)

The best validation mse is 0.420326, really bad performance compared with the baseline model.

#### 3.2 RNN Model
##### 3.2.1 Bidrectional GRU with spatial dropout
```python
def get_model():
    inp = kl.Input(shape=(maxlen,))
    embed = kl.Embedding(max_features,embed_size,weights=[embedding_matrix],trainable=False)(inp)
    embed = kl.SpatialDropout1D(0.2)(embed)
    gru = kl.Bidirectional(kl.CuDNNGRU(128,return_sequences=True))(embed)
    maxpool = kl.GlobalMaxPool1D()(gru)
    avgpool = kl.GlobalAvgPool1D()(gru)
    out = kl.concatenate([maxpool,avgpool])
    out = kl.Dense(1)(out)
    model = ks.models.Model(inp,out)
    model.compile(loss="mse", optimizer='adam', metrics=[])
    return model
```
![gru_with_spatial_dropout](http://ok669z6cd.bkt.clouddn.com/gru_spatialdrop_static.png)

The best validation mse is 0.406670, slightly worse than the baseline model.
