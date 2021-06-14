#basic imports
import numpy as np
import os
import cv2
import tensorflow as tf

CATEGORIES=["0","1","2","3","4","5","6","7","8","9",
           "adna","ba","bha","cha","chha","chhya","da","daa","dha","dhaa","ga",
           "gha","gya","ha","ja","jha","ka","kha","kna","ksha","la","ma","na",
           "pa","pha","ra","sa","sh","t","ta","tha","thaa","tra","waw","yaw","yna"]


# use if trained on grayscale specified: model-2
def prepare(filepath):
    img_array=cv2.imread(filepath,cv2.IMREAD_GRAYSCALE)
    new_array=cv2.resize(img_array,(32,32))
    image=np.array(new_array).reshape((-1, 32, 32, 1))
    
    return image 

# use if trained on grayscale not specified: model-1
def prepare(filepath):
    img_array=cv2.imread(filepath)
    new_array=cv2.resize(img_array,(32,32))
    image=np.array(new_array).reshape((-1, 32, 32, 3))
    
    return image 




model=tf.keras.models.load_model('devnagri-script-detection-2.model')


prediction=[model.predict_classes([prepare('1.jpeg')])]
prediction.append(model.predict_classes([prepare('2.jpeg')]))
prediction.append(model.predict_classes([prepare('3.jpeg')]))
prediction.append(model.predict_classes([prepare('4.jpeg')]))
prediction.append(model.predict_classes([prepare('5.jpeg')]))
prediction.append(model.predict_classes([prepare('6.jpeg')]))
prediction.append(model.predict_classes([prepare('7.jpeg')]))



for l in range(len(prediction)):
    print((l+1),"  ",CATEGORIES[int(prediction[l])], "  ", prediction[l])
'''

print(CATEGORIES[int(prediction[0])])
'''