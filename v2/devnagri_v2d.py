#basic imports

import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import random
import tensorflow as tf
from tensorflow.keras.models import Sequential
import tensorflow.keras.layers as tfl



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
            img_array=cv2.imread(os.path.join(path,img),cv2.IMREAD_GRAYSCALE)
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

X=np.array(X).reshape((-1,32,32,1)) # .reshape(-1,32,32,1)
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

model.save('devnagri-script-detection-2.model')