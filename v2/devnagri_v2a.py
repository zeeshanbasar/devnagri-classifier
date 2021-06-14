#basic imports

import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import pickle
import random
import tensorflow as tf



# load the directory where image(or dataset) folders are
DATADIR='C:\IDK\ML\Devnagri\DevanagariHandwrittenCharacterDataset\Train'
CATEGORIES=["0","1","2","3","4","5","6","7","8","9",
           "adna","ba","bha","cha","chha","chhya","da","daa","dha","dhaa","ga",
           "gha","gya","ha","ja","jha","ka","kha","kna","ksha","la","ma","na",
           "pa","pha","ra","sa","sh","t","ta","tha","thaa","tra","waw","yaw","yna"]


# creating the training data from the images dataset
training_data=[]
def create_training_data():
    for category in CATEGORIES:
        path=os.path.join(DATADIR,category)
        class_num=CATEGORIES.index(category)
        for img in os.listdir(path):
            img_array=cv2.imread(os.path.join(path,img))
            new_array=cv2.resize(img_array,(32,32))            
            image = tf.cast(new_array, tf.float32)
            training_data.append([image,class_num])
create_training_data()


random.shuffle(training_data)


X=[]
y=[]

for features, labels in training_data:
	X.append(features)
	y.append(labels)

X=np.array(X).reshape(-1,32,32,1)


# to save X,y as proper training data
pickle_out=open("X.pickle","wb")
pickle.dump(X,pickle_out)
pickle_out.close()

pickle_out=open("y.pickle","wb")
pickle.dump(y,pickle_out)
pickle_out.close()