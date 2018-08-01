import numpy as np
np.random.seed(2018)
import pandas as pd
import matplotlib.pyplot as plt
import os
import re
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.externals import joblib

from keras.callbacks import Callback
import keras as ks
import keras.layers as kl
import gensim

from models import get_model

features = joblib.load("./tempdata/features.npy")
labels = joblib.load("./tempdata/labels.npy")

x_train, x_valid, y_train, y_valid = train_test_split(features,labels,
	test_size=0.4,shuffle=True,random_state=2018)

model = get_model()
early_stopping = ks.callbacks.EarlyStopping(monitor='val_loss',
                              min_delta=0,
                              patience=3,
                              verbose=0, mode='auto')
history = model.fit(x_train,y_train,validation_data=(x_valid,y_valid),batch_size=128,
      epochs=20,callbacks=[early_stopping])

print(history.history['loss'])
print(history.history['val_loss'])
num_epoch = len(history.history['loss'])
plt.plot(range(1,num_epoch+1),history.history['loss'])
plt.plot(range(1,num_epoch+1),history.history['val_loss'])
plt.scatter(range(1,num_epoch+1),history.history['loss'])
plt.scatter(range(1,num_epoch+1),history.history['val_loss'])
plt.ylabel('MSE')
plt.xlabel('epoch')
plt.legend(['train mse', 'test mse'], loc='lower left');
