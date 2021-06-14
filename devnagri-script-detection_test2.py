# basic imports
import numpy as np
#import matplotlib.pyplot as plt
#import os
#import cv2
import pickle
import tensorflow as tf
from tensorflow.keras.models import Sequential
import tensorflow.keras.layers as tfl
#from tensorflow.keras.callbacks import TensorBoard, LearningRateScheduler, EarlyStopping
#import time


#model name
#NAME="devnagri-{}".format(int(time.time()))

# TensorBoard
#tensorboard=TensorBoard(log_dir='logs/{}'.format(NAME))

# GPU
gpu_options=tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.333)
sess=tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))

# load the directory where image(or dataset) folders are

DATADIR='C:/IDK/ML/Devnagri/DevanagariHandwrittenCharacterDataset/Train'
CATEGORIES=["0","1","2","3","4","5","6","7","8","9",
           "adna","ba","bha","cha","chha","chhya","da","daa","dha","dhaa","ga",
           "gha","gya","ha","ja","jha","ka","kha","kna","ksha","la","ma","na",
           "pa","pha","ra","sa","sh","t","ta","tha","thaa","tra","waw","yaw","yna"]


X=pickle.load(open("X.pickle","rb"))
y=pickle.load(open("y.pickle","rb"))


#X=np.array(X).reshape((-1,32,32,1))
y=np.array(y)
y=tf.keras.utils.to_categorical(y, num_classes=46, dtype='float32')


#building the model

model=Sequential()
# layer 1
model.add(tfl.Conv2D(64,(3,3)))
model.add(tfl.Activation('relu'))
model.add(tfl.MaxPooling2D(pool_size=(2,2)))
#layer 2
model.add(tfl.Conv2D(64,(3,3)))
model.add(tfl.Activation('relu'))
model.add(tfl.MaxPooling2D(pool_size=(2,2)))
#layer 3
model.add(tfl.Conv2D(64,(3,3)))
model.add(tfl.Activation('relu'))
model.add(tfl.MaxPooling2D(pool_size=(2,2)))
# Dense 1
model.add(tfl.Flatten())
model.add(tfl.Dense(64))
model.add(tfl.Activation('relu'))
# Dense 2
model.add(tfl.Flatten())
model.add(tfl.Dense(64))
model.add(tfl.Activation('relu'))
# o/p layer
model.add(tfl.Dense(46))        # 1 for binary class
model.add(tfl.Activation('softmax'))

# compilation, loss, accuracy, optimizer
model.compile(loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
             optimizer=tf.keras.optimizers.Adam(),metrics=['accuracy','AUC'])


# early=tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.0000001, patience=5)

# fitting
model.fit(X,y,validation_split=0.1,batch_size=200,epochs=10)#,callbacks=[tensorboard,early])

model.save('devnagri-script-detection.model')
