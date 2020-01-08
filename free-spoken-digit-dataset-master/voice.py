# -*- coding: utf-8 -*-
"""
Created on Sat Dec 28 16:28:59 2019

@author: Ram Kumar R P
"""
import librosa
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf

wav, sr = librosa.load('D:\\Project\\free-spoken-digit-dataset-master\\free-spoken-digit-dataset-master\\recordings\\0_nicolas_14.wav')
print ('sr:', sr)
print ('wav shape:', wav.shape)
print ('length:', sr/wav.shape[0], 'secs')
plt.plot(wav)
plt.plot(wav[4000:4200]) 

DATA_DIR = 'D:\\Project\\free-spoken-digit-dataset-master\\free-spoken-digit-dataset-master\\recordings\\'
X = []
y = []
pad = lambda a, i: a[0: i] if a.shape[0] > i else np.hstack((a, np.zeros(i - a.shape[0])))
for fname in os.listdir(DATA_DIR):
    struct = fname.split('_')
    digit = struct[0]
    wav, sr = librosa.load(DATA_DIR + fname)
    padded = pad(wav, 30000)
    X.append(padded)
    y.append(digit)
X = np.vstack(X)
y = np.array(y)



ip = tf.keras.layers.Input(shape=(X[0].shape))
hidden = tf.keras.layers.Dense(128, activation='relu')(ip)
op = tf.keras.layers.Dense(10, activation='softmax')(hidden)
model = tf.keras.Model(ip, op)
model.summary()

from sklearn.model_selection import train_test_split

train_X, test_X, train_y, test_y = train_test_split(X,y,test_size = 0.2,random_state=0)
from keras.utils import to_categorical
test_y = to_categorical(test_y)
from keras.utils import to_categorical
train_y = to_categorical(train_y)
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
history = model.fit(train_X,
          train_y,
          epochs=10,
          batch_size=32,
          validation_data=(test_X, test_y))

model.predict(test_X[0])
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
