# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 18:02:54 2019

@author: Arham Jain
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from PIL import Image

dataset=pd.read_csv('train.csv')
import keras
from keras.datasets import mnist
(x_train,y_train),(x_test,y_test)=mnist.load_data()
plt.imshow(x_train[1],cmap='gray')
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

dataset=np.array(dataset)
test_ydata=dataset[:,0]
test_ydata = keras.utils.to_categorical(test_ydata, 10)
dataset=dataset[:,1:785].reshape((42000,28,28))
dataset=dataset/255
plt.imshow(dataset[5],cmap='gray')
x_train=x_train.reshape((60000,28,28,1))
x_train=x_train/255
x_test=x_test.reshape((10000,28,28,1))
x_test=x_test/255

from sklearn.model_selection import train_test_split
x_train, x_cross, y_train, y_cross = train_test_split(x_train, y_train, test_size = 1/6, random_state = 1)


from keras.models import Sequential
from keras.layers import Convolution2D as conv2d
from keras.layers import MaxPooling2D as pool2d
from keras.layers import Flatten as flatten
from keras.layers import Dense as dense

model=Sequential()

model.add(conv2d(500, 3, 3, input_shape = (28, 28,1), activation = 'relu'))
model.add(pool2d(pool_size=(2,2),strides=(1,1)))

model.add(conv2d(400, 3, 3, activation = 'relu'))
model.add(pool2d(pool_size=(2,2),strides=(1,1)))

model.add(conv2d(300, 3, 3, activation = 'relu'))
model.add(pool2d(pool_size=(2,2),strides=(1,1)))

model.add(conv2d(200, 3, 3, activation = 'relu'))
model.add(pool2d(pool_size=(2,2),strides=(1,1)))

model.add(conv2d(100, 3, 3, activation = 'relu'))
model.add(pool2d(pool_size=(2,2),strides=(1,1)))



model.add(flatten())

model.add(dense(output_dim = 128, activation = 'relu'))

model.add(dense(output_dim = 10, activation = 'sigmoid'))


model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

#from keras.preprocessing.image import ImageDataGenerator
#
#datagen = ImageDataGenerator(
#    featurewise_center=True,
#    featurewise_std_normalization=True,
#    rotation_range=20,
#    width_shift_range=0.2,
#    height_shift_range=0.2,
#    horizontal_flip=False)
#
## compute quantities required for featurewise normalization
## (std, mean, and principal components if ZCA whitening is applied)
#datagen.fit(x_train)
#
#
## fits the model on batches with real-time data augmentation:
#model.fit_generator(datagen.flow(x_train, y_train, batch_size=128),
#                    steps_per_epoch=len(x_train) / 128, epochs=12,validation_data = test_set)



model.fit(x_train, y_train,
          batch_size=128,
          epochs=12,
          verbose=1,
          validation_data=(x_cross, y_cross))

from keras.models import load_model
model.save_weights('modelwith98.79.h5')
model.save("modelwith99.h5")
load=load_model("modelwith99.h5")
quest=pd.read_csv('quest.csv')
quest=np.array(quest)
quest=quest.reshape((28000,28,28,1))
answer=load.predict(quest)
ans=[]


ans=np.argmax(answer)
load.evaluate(x_test,y_test) 
model.evaluate(x_test,y_test)


def predictnumber(x):

    x=x.reshape((1,28,28,1))
    out=model.predict(x)
    ans=np.argmax(out)
    x=x.reshape((28,28))
    plt.imshow(x,cmap='gray')
    print(ans,end="")
    return 
predictnumber(x_test[9190])
def segmented_predict(filename):
    t = 200

# read original image
    image = cv2.imread(filename)
#image=cv2.bitwise_not(image)


# create binary image
    gray = cv2.cvtColor(src = image, code = cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(src = gray,
                            ksize = (5, 5),
                            sigmaX = 0)
    (t, binary) = cv2.threshold(src = gray,
        thresh = t,
        maxval = 255,
        type = cv2.THRESH_BINARY)
    seg=binary.copy()
    binary=cv2.bitwise_not(binary)

    contours, hierarchy = cv2.findContours(binary,  
                                           cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for i in range(len(contours)):
        x=contours[i][:,0,:]

        new_image=seg[:,min(x[:,0])-5:max(x[:,0])+5]
        img = Image.fromarray(new_image, 'L')
        
        img = img.resize((28,28), Image.ANTIALIAS)
        plt.imshow(img,cmap='gray')
        new_image=np.array(img)
        (t, binary1) = cv2.threshold(src = new_image,
                                    thresh = 200,
                                    maxval = 255,
                                    type = cv2.THRESH_BINARY)
        
        predictnumber(binary1)
    return
filename='digits.png'
segmented_predict(filename)










