# basic imports
import numpy as np
import random
import math
import os
import cv2
import pickle
import tensorflow as tf
from tensorflow.keras.models import Sequential
import tensorflow.keras.layers as tfl
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
from keras.preprocessing.image import img_to_array, load_img
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.inception_v3 import preprocess_input
import time


# #model name
# NAME="devnagri-{}".format(int(time.time()))

# # TensorBoard
# tensorboard=TensorBoard(log_dir='logs/{}'.format(NAME))

# GPU
gpu_options=tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.333)
sess=tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))


X=pickle.load(open("X.pickle","rb"))
y=pickle.load(open("y.pickle","rb"))



dense_layers=[4]
layer_sizes=[256]
conv_layers=[4]
for dense_layer in dense_layers:
    for layer_size in layer_sizes:
        for conv_layer in conv_layers:
            # TensorBoard
            NAME="{}-conv-{}-nodes-{}-dense-model-A".format(conv_layer,layer_size,dense_layer)
            tensorboard=TensorBoard(log_dir='logs/{}'.format(NAME))
            print(NAME)
            
            model=Sequential()
            # layer 1
            model.add(tfl.Conv2D(layer_size,(3,3),padding='same'))
            model.add(tfl.Activation('relu'))
            model.add(tfl.MaxPool2D(pool_size=(2,2)))
            
            # next predetermined conv layers in conv_layers
            
            for l in range(conv_layer-1):
                #layer l-1
                model.add(tfl.Conv2D(layer_size,(3,3),padding='same'))
                model.add(tfl.Activation('relu'))
                model.add(tfl.MaxPool2D(pool_size=(2,2)))
            
            model.add(tfl.Flatten()) # to flatten out the o/p of last conv layer, basically only one is needed
            
            # dense layers as predetermined in dense layer
            
            for l in range(dense_layer):
                model.add(tfl.Dense(layer_size))
                model.add(tfl.Activation('relu'))
            
            # o/p layer
            model.add(tfl.Dense(46))        # 1 for binary class
            model.add(tfl.Activation('softmax'))

            early=tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.0000001, patience=5)

            # compilation, loss, accuracy, optimizer
            model.compile(loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
            	optimizer=tf.keras.optimizers.Adam(),metrics=['accuracy','AUC'])
            # fitting
            model.fit(X,y,validation_split=0.1,batch_size=50,epochs=1000,callbacks=[tensorboard,early])


# #building the model
# model=Sequential()
# # layer 1
# model.add(tfl.Conv2D(64,(3,3)))
# model.add(tfl.Activation('relu'))
# model.add(tfl.MaxPooling2D(pool_size=(2,2)))
# #layer 2
# model.add(tfl.Conv2D(64,(3,3)))
# model.add(tfl.Activation('relu'))
# model.add(tfl.MaxPooling2D(pool_size=(2,2)))
# #layer 3
# model.add(tfl.Conv2D(64,(3,3)))
# model.add(tfl.Activation('relu'))
# model.add(tfl.MaxPooling2D(pool_size=(2,2)))
# # Dense 1
# model.add(tfl.Flatten())
# model.add(tfl.Dense(64))
# model.add(tfl.Activation('relu'))
# # Dense 2
# model.add(tfl.Flatten())
# model.add(tfl.Dense(64))
# model.add(tfl.Activation('relu'))
# # o/p layer
# model.add(tfl.Dense(46))        # 1 for binary class
# model.add(tfl.Activation('softmax'))

# # compilation, loss, accuracy, optimizer
# model.compile(loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
#              optimizer=tf.keras.optimizers.Adam(),metrics=['accuracy','AUC'])


# early=tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.0000001, patience=5)

# # fitting
# model.fit(X,y,validation_split=0.1,batch_size=20,epochs=200,callbacks=[tensorboard,early])

model.save('devnagri-script-detection-3-main.model')