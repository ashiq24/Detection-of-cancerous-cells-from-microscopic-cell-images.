# -*- coding: utf-8 -*-
"""
Created on Mon Jan  7 13:08:18 2019

@author: User
"""
import keras
from keras.layers.merge import Concatenate
import numpy as np
from keras import optimizers
from keras.layers import Input,Dense,Flatten,Dropout,merge,Reshape,Conv2D,MaxPooling2D,UpSampling2D,Conv2DTranspose
from keras.layers.normalization import BatchNormalization
from keras.models import Model,Sequential
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adadelta, SGD,Adam
from keras import regularizers
from keras import backend as K
from keras.utils import to_categorical
import random
from keras.applications.nasnet import NASNetMobile
from keras import applications
from keras.preprocessing import image
from keras.applications import densenet
from keras.layers import GlobalAveragePooling2D
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from PIL import Image
import numpy as np
import cv2 as cv
import glob

from scipy.signal import convolve2d
random.seed(20)

def edge(im_small):
  n=100
  sobel_x = np.c_[
      [-1,0,1],
      [-2,0,2],
      [-1,0,1]
  ]

  sobel_y = np.c_[
      [1,2,1],
      [0,0,0],
      [-1,-2,-1]
  ]

  ims = []
  for d in range(3):
      sx = convolve2d(im_small[:,:,d], sobel_x, mode="same", boundary="symm")
      sy = convolve2d(im_small[:,:,d], sobel_y, mode="same", boundary="symm")
      ims.append(np.sqrt(sx*sx + sy*sy))

  im_conv = np.stack(ims, axis=2).astype("uint8")
  return im_conv

def emboss(im_small):
  karnel = np.array(
  [
      [-1,-1,0],
      [-1,0,1],
      [0,1,1]
  ])
  ims = []
  for d in range(3):
      sx = convolve2d(im_small[:,:,d], karnel, mode="same", boundary="symm")
      ims.append(sx)
  im_conv = np.stack(ims, axis=2).astype("uint8")
  return im_conv



Images = []
labels = []
dim = 128
from scipy import ndimage
alpha = 300
#print(glob.glob('/content/gdrive/My Drive/train/fold_1/all/*.bmp'))
m=0

for filename in glob.glob('fold_0/all/*.bmp'): #assuming gif
    m+=1
    
    im = image.load_img(filename, target_size=(dim, dim))
    im = image.img_to_array(im)
    im = np.array(im)
    
    filter_blurred_f = ndimage.gaussian_filter(im, 1)
    sharpened = im + alpha * (im - filter_blurred_f)
    im = emboss(im)
    #im = edge(im)
    #im = edge(im)
    Images.append(im)
    labels.append(1)
    print(m)
m=0
for filename in glob.glob('fold_1/all/*.bmp'): #assuming gif
    m+=1
    
    try:
      im = image.load_img(filename, target_size=(dim, dim))
      im = image.img_to_array(im)
      im = np.array(im)
      filter_blurred_f = ndimage.gaussian_filter(im, 1)
      sharpened = im + alpha * (im - filter_blurred_f)
      im = emboss(im)
      #im = edge(im)
      #im = edge(im)
      Images.append(im)
      labels.append(1)
    except(OSError):
      print('error',m)
print(m)

m=0    
for filename in glob.glob('fold_0/hem/*.bmp'): #assuming gif
    m+=1
    
    im = image.load_img(filename, target_size=(dim, dim))
    im = image.img_to_array(im)
    im = np.array(im)
    filter_blurred_f = ndimage.gaussian_filter(im, 1)
    sharpened = im + alpha * (im - filter_blurred_f)
    im = emboss(im)
    #im = edge(im)
    #im = edge(im)
    Images.append(im)
    labels.append(0)
print(m)
m=0
for filename in glob.glob('fold_1/hem/*.bmp'): #assuming gif
    m+=1
    
    im = image.load_img(filename, target_size=(dim, dim))
    im = image.img_to_array(im)
    im = np.array(im)
    filter_blurred_f = ndimage.gaussian_filter(im, 1)
    sharpened = im + alpha * (im - filter_blurred_f)
    im = emboss(im)
    #im = edge(im)
    #im = edge(im)
    Images.append(im)
    labels.append(0)
print(m)
m=0


print(len(Images), Images[0].shape)
import matplotlib.pyplot as plt
plt.imshow( image.array_to_img(Images[7]) , cmap='gray')
def get_model_fusion(automodel):
    # create a placeholder for an encoded (32-dimensional) input
    #input_img = Input(shape=(75, 75, 3), name="input_img")
    input_img = automodel.get_layer('input_img').output
    #input_2 = Input(shape=[1], name="angle")
    # retrieve the last layer of the autoencoder model
    encoder_out = automodel.get_layer('encoded').output
    # 10* 10
    #img_1 = GlobalMaxPooling2D() (encoder_out)
    img_1 = GlobalAveragePooling2D() (encoder_out)
    #add global poool ??
    #img_concat =  (Concatenate()([img_1,input_2 ]))
    dense_layer = Dense(512, activation='relu', name='fcc1')(img_1 )
    #dense_layer = BatchNormalization(momentum=0)(dense_layer )
    dense_layer = Dropout(0.2)(dense_layer)
    dense_layer = Dense(256, activation='relu', name='fcc2')(dense_layer)
    #dense_layer = BatchNormalization(momentum=0)(dense_layer )
    dense_layer = Dropout(0.2)(dense_layer)    
    predictions = Dense(1, activation='sigmoid')(dense_layer)
    
    #model = Model(input=base_model.input, output=predictions)
    model = Model(input_img, predictions)
    optimizer = Adam(lr=1e-3, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    #optimizer=RMSprop(lr=0.001)
    model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"])
    return model

def get_model_autoencoder():
    
    input_img = Input(shape=(dim, dim, 3), name="input_img")  # adapt this if using `channels_first` image data format
    bn_model = 0
    x = Conv2D(16, (3, 3), activation='elu', padding='same',kernel_initializer='random_uniform')((BatchNormalization(momentum=bn_model))(input_img))
    x = Conv2D(16, (3, 3), activation='elu', padding='same',kernel_initializer='random_uniform')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Dropout(0.3)(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same',kernel_initializer='random_uniform')(x)
    x = Conv2D(32, (3, 3), activation='elu', padding='same',kernel_initializer='random_uniform')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Dropout(0.3)(x)
    x = Conv2D(64, (3, 3), activation='elu', padding='same',kernel_initializer='random_uniform')(x)
    x = Conv2D(64, (3, 3), activation='elu', padding='same',kernel_initializer='random_uniform')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Dropout(0.3)(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same',kernel_initializer='random_uniform')(x)
    encoded = MaxPooling2D((2, 2), padding='same', name="encoded")(x)

    x = Conv2D(128, (3, 3), activation='elu', padding='same',kernel_initializer='random_uniform')(encoded)
    x = Dropout(0.3)(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='elu', padding='same',kernel_initializer='random_uniform')(x)
    x = Conv2D(64, (3, 3), activation='elu', padding='same',kernel_initializer='random_uniform')(x)
    x = Dropout(0.3)(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(32, (3, 3), activation='elu', padding='same',kernel_initializer='random_uniform')(x)
    x = Conv2D(32, (3, 3), activation='elu', padding='same',kernel_initializer='random_uniform')(x)
    x = Dropout(0.3)(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(16, (3, 3), activation='elu', padding='same',kernel_initializer='random_uniform')(x)
    x = Conv2D(16, (3, 3), activation='elu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(3, (2, 2), activation='sigmoid', padding='same',name="decoded")(x)
    
    autoencoder = Model(input_img, decoded)
    optimizer = Adam(lr=1e-6, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    autoencoder.compile(optimizer=optimizer, loss='binary_crossentropy',)
    return autoencoder



    
indices=list(range(len(Images)))
np.random.seed(23)
np.random.shuffle(indices)
Images = np.array(Images)
Images= Images/255.0
labels = np.array(labels)
#labels=np_utils.to_categorical(labels)
ind=int(len(indices)*0.60)
ind2 = int(len(indices)*0.20)
# train data
X_train=Images[indices[:ind]] 
y_train=labels[indices[:ind]]
# validation data
X_val=Images[indices[ind:ind+ind2]] 
y_val=labels[indices[ind:ind+ind2]]

X_test = Images[indices[ind+ind2:]] 
y_test = labels[indices[ind+ind2:]]

print(X_train[0].shape)

image_gen = ImageDataGenerator(
    samplewise_center=True,
    featurewise_std_normalization=True,
    featurewise_center=True,
   
    zca_epsilon=0.7,
    zca_whitening=True,
    )

model= get_model_autoencoder()
model.summary()
    
model.fit(x=X_train, y=X_train,
            epochs=5, 
            verbose=1, 
            validation_data=(X_val,X_val),
            shuffle=True,
            
            )



## AUTO ENCODING FINISHED 



#predictions=model.predict(X_train,verbose=1)

funsion_model = get_model_fusion(model)

from keras.callbacks import ModelCheckpoint

checkpointer = ModelCheckpoint('model.h5', verbose=1, save_best_only=True)
funsion_model.summary()

funsion_model.fit(x=X_train, y=y_train,
            epochs=5, 
            verbose=1, 
            validation_data=(X_val,y_val),
            shuffle=True,
            callbacks=[
                checkpointer,
            ]
            )

funsion_model.evaluate(X_test, y_test)



















