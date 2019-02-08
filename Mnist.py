# -*- coding: utf-8 -*-
"""
Created on Thu Feb  7 20:56:11 2019

@author: sai kumar naik
"""

from __future__ import print_function

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.optimizers import RMSprop
from keras.layers import Conv2D, MaxPooling2D

batch_size = 128
num_classes = 10
epochs = 20

# the data, shuffled and split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

#Please construct the following neural network and report accuracy after training
#Layer 1: 2D Convolution with 32 hidden neurons, kernel 3 by 3, relu activation, input_shape (28,28,1)
#Layer 2: 2D MaxPooling, pool_size 2 by 2
#Layer 3: Flatten (Hint: model.add(Flatten()))
#Layer 4 Softmax Output Layer according to the problem (output vector)

model=Sequential()

model.add(Dense(512,activation="relu",input_shape=(784,)))
model.add(Dropout(0.2))
model.add(Dense(300,activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(150,activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(100,activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(num_classes,activation="softmax"))
model.summary()

model.compile(loss="categorical_crossentropy",optimizer="sgd",metrics=['accuracy'])
#some learners constantly reported 502 errors in Watson Studio. 
#This is due to the limited resources in the free tier and the heavy resource consumption of Keras.
#This is a workaround to limit resource consumption

#from keras import backend as K

#K.set_session(K.tf.Session(config=K.tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)))

history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])




