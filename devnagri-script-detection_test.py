#basic imports

import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import tensorflow as tf
from tensorflow.keras.models import Sequential
import tensorflow.keras.layers as tfl
from tensorflow.keras.callbacks import TensorBoard
import time
import pickle
import random

#model name
# NAME="devnagri-{}".format(int(time.time()))

# TensorBoard
# tensorboard=TensorBoard(log_dir='logs/{}'.format(NAME))

# GPU
gpu_options=tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.333)
sess=tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))

# load the directory where image(or dataset) folders are

DATADIR='C:/IDK/ML/Devnagri/DevanagariHandwrittenCharacterDataset/Train'
CATEGORIES=["0","1","2","3","4","5","6","7","8","9",
           "adna","ba","bha","cha","chha","chhya","da","daa","dha","dhaa","ga",
           "gha","gya","ha","ja","jha","ka","kha","kna","ksha","la","ma","na",
           "pa","pha","ra","sa","sh","t","ta","tha","thaa","tra","waw","yaw","yna"]

# creating the training data from the images dataset
IMG_SIZE=32
training_data=[]
def create_training_data():
    for category in CATEGORIES:
        path=os.path.join(DATADIR,category)
        class_num=CATEGORIES.index(category)
        for img in os.listdir(path):
            try:
                img_array=cv2.imread(os.path.join(path,img))
                image = tf.cast(img_array, tf.float32)
                training_data.append([image,class_num])
            except Exception as e:
                pass
create_training_data()


#random shuffle
random.shuffle(training_data)


# separate out features and labels

X=[]
y=[]

for features, labels in training_data:
    X.append(features)
    y.append(labels)
    
#X=np.array(X).reshape(-1,IMG_SIZE,IMG_SIZE,1)
#y=np.array(y)


# to save X,y as proper training data

import pickle

pickle_out=open("X.pickle","wb")
pickle.dump(X,pickle_out)
pickle_out.close()

pickle_out=open("y.pickle","wb")
pickle.dump(y,pickle_out)
pickle_out.close()