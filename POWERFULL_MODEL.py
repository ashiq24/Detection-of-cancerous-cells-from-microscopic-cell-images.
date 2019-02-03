###IMPORTANT NOTICE
'''
this code is run in Colab Google. So, this code takes folder from drive.
So if you run it in your pc you need not run the code to mount the drive.

For this code only please use the link below to download the data:

MAIN_TRAIN = https://drive.google.com/drive/folders/1nyXFxeP9Oaf3MDg-uUtKroJ1iR8pZ7yV?fbclid=IwAR27WzAsNHwdMVLppzEAyxt5UppaeYSCb3jmlnJAa90APtkUEfUTJkpPgMM

MAIN_VALIDATION = https://drive.google.com/drive/folders/1GKwrHFAl0wFtvuQ3ZmxbHhg7aFryq6wU?fbclid=IwAR2T12Q1C0GqwGr79jt0LwphMguEsxEVprOzF0dKllE33kzj-U0SefO28Os 


And all the folder links are given with respect to the drive. So , if you want to run this code in your machine please 
change the path accrodingly.
For example :
if drive path is : '/content/gdrive/My Drive/train/fold_0/all/*.bmp'
then for your machine it will be  : 'YOUR_FOLDER_NAME/train/fold_0/all/*.bmp'



As this model is very powerful and it was over fitting the data. so didn't go into deep and train
this model in full using train , test and validation set .



'''

from google.colab import drive
drive.mount('/content/gdrive')

from keras import applications
from keras import optimizers
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D,Flatten,Dropout
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from PIL import Image
import glob
import numpy as np
import cv2 as cv

from scipy.signal import convolve2d
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

def simple_threshold(im, threshold=50):
    return ((im > threshold) * 255).astype("uint8")

def sifat(image):
    original_image = image

    #contrast change
    new_image = np.zeros(image.shape, image.dtype)
    alpha = 4.4 # Simple contrast control
    beta = -10    # Simple brightness control
    # Initialize values
    #print(' Basic Linear Transforms ')
    print('-------------------------')

    # Do the operation new_image(i,j) = alpha*image(i,j) + beta
    # Instead of these 'for' loops we could have used simply:
    # new_image = cv.convertScaleAbs(image, alpha=alpha, beta=beta)
    # but we wanted to show you how to access the pixels :)
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            for c in range(image.shape[2]):
                new_image[y,x,c] = np.clip(alpha*image[y,x,c] + beta, 0, 255)



    image = new_image
    #brightness with gamma correction
    gamma = 0.8
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")

    # apply gamma correction using the lookup table
    new_image = cv.LUT(image, table)



    image = new_image
    #Sharpening
    # Create our shapening kernel, it must equal to one eventually
    kernel_sharpening = np.array([[-1,-1,-1],
                                  [-1, 9,-1],
                                  [-1,-1,-1]])
    # applying the sharpening kernel to the input image & displaying it.
    new_image = cv.filter2D(image, -1, kernel_sharpening)



    image = new_image
    #Saturation change
    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)

    for i in range(len(hsv)):
        for j in range(len(hsv[i])):
            hsv[i,j,1] += 75 #change saturation value
            #hsv[i,j,0] += 0 #change hue value
            #hsv[i,j,2] += 10

    new_image = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
    return new_image

Images = []
labels = []
dim = 96
from scipy import ndimage
alpha = 300
#print(glob.glob('/content/gdrive/My Drive/train/fold_1/all/*.bmp'))
m=0

for filename in glob.glob('/content/gdrive/My Drive/maintrain/all/*.bmp'): #assuming gif
    m+=1
    im = image.load_img(filename, target_size=(dim, dim))
    im = image.img_to_array(im)
    im = np.array(im)
    
    filter_blurred_f = ndimage.gaussian_filter(im, 1)
    sharpened = im + alpha * (im - filter_blurred_f)
    
    im = edge(im)
    im = emboss(im)
    #im = edge(im)
    Images.append(im)
    labels.append(1)
    print(m)
m=0
for filename in glob.glob('/content/gdrive/My Drive/mainvalidation/all/*.bmp'): #assuming gif
    m+=1
    try:
      im = image.load_img(filename, target_size=(dim, dim))
      im = image.img_to_array(im)
      im = np.array(im)
      filter_blurred_f = ndimage.gaussian_filter(im, 1)
      sharpened = im + alpha * (im - filter_blurred_f)
      
      im = edge(im)
      im = emboss(im)
      #im = edge(im)
      Images.append(im)
      labels.append(1)
    except(OSError):
      print('error',m)
print(m)
m=0

for filename in glob.glob('/content/gdrive/My Drive/maintrain/hem/*.bmp'): #assuming gif
    m+=1
    im = image.load_img(filename, target_size=(dim, dim))
    im = image.img_to_array(im)
    im = np.array(im)
    filter_blurred_f = ndimage.gaussian_filter(im, 1)
    sharpened = im + alpha * (im - filter_blurred_f)
    
    im = edge(im)
    im = emboss(im)
    #im = edge(im)
    Images.append(im)
    labels.append(0)
print(m)
m=0
for filename in glob.glob('/content/gdrive/My Drive/mainvalidation/hem/*.bmp'): #assuming gif
    m+=1
    im = image.load_img(filename, target_size=(dim, dim))
    im = image.img_to_array(im)
    im = np.array(im)
    filter_blurred_f = ndimage.gaussian_filter(im, 1)
    sharpened = im + alpha * (im - filter_blurred_f)
    
    im = edge(im)
    im = emboss(im)
    #im = edge(im)
    Images.append(im)
    labels.append(0)
print(m)

print(len(Images), Images[0].shape)
import matplotlib.pyplot as plt
plt.imshow( image.array_to_img(Images[7]) , cmap='gray')
import pickle
with open('Images.pkl', 'wb') as output:
    pickle.dump(Images, output, pickle.HIGHEST_PROTOCOL)
with open('labels.pkl', 'wb') as output:
    pickle.dump(labels, output, pickle.HIGHEST_PROTOCOL)

with open('Images.pkl', 'rb') as input:
    Images = pickle.load(input)
with open('labels.pkl', 'rb') as input:
    labels = pickle.load(input)

from keras.applications.vgg16 import VGG16
from keras.applications.nasnet import NASNetMobile
from keras.applications.mobilenet_v2 import MobileNetV2
from keras.applications.densenet import DenseNet121
from keras.applications.nasnet import NASNetMobile
from keras.applications.mobilenet_v2 import MobileNetV2

def nin_cnn(k,model_input):#k=1 means resnet50 model, otherwise k=2 inceptionv3,k=3 vgg can extend it for many models with choosing value of k
# create the base pre-trained model
    if(k==1):
        base_model =MobileNetV2(weights='imagenet', include_top=False,input_shape=(96,96,3))
    elif(k==2):
        base_model =DenseNet121(weights='imagenet', include_top=False,input_shape=(96,96,3))
    elif(k==3):
        base_model =NASNetMobile(weights='imagenet', include_top=False,input_shape=(96,96,3))
# add a global spatial average pooling layer
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
# let's add a fully-connected layer
    x= Dropout(0.4)(x)
    x = Dense(100, activation='relu')(x)
    x= Dropout(0.25)(x)
    
   
# and a logistic layer -- let's say we have 200 classes
    predictions = Dense(1, activation='sigmoid')(x)

# this is the model we will train
    if(k==1):
        model = Model(inputs=base_model.input, outputs=predictions,name='mobilenet')
    elif(k==2):
        model = Model(inputs=base_model.input, outputs=predictions,name='densenet')
    elif(k==3):
        model = Model(inputs=base_model.input, outputs=predictions,name='nasnet')

    for layer in base_model.layers:
        layer.trainable = True
    return model

def ensemble(models, model_input):
    
    outputs = [model.outputs[0] for model in models]
    y = Average()(outputs)#can use Average instead of maximum , need to see which one performs better
    
    model = Model(model_input, y, name='ensemble')
    
    return model

def buildmodel(model_input):#k=1 means resnet50 model, otherwise k=2 inceptionv3,k=3 vgg can extend it for many models with choosing value of k
# create the base pre-trained model
   
    res_model =MobileNetV2(weights='imagenet', include_top=False)
    x = res_model(model_input)
    layer1 = GlobalAveragePooling2D()(x)
# let's add a fully-connected layer
    
    inception_model =DenseNet121(weights='imagenet', include_top=False)
    x = inception_model(model_input)
    layer2=GlobalAveragePooling2D()(x)
    
    vgg16_model = MobileNetV2(weights=None, include_top=False)
    x = vgg16_model(model_input)
    layer3 = GlobalAveragePooling2D()(x)
    
    x=keras.layers.concatenate([layer1,layer2,layer3])
    x= Dropout(0.4)(x)
    x = Dense(100, activation='relu')(x)
    x= Dropout(0.25)(x)
# and a logistic layer -- let's say we have 200 classes
    predictions = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=model_input, outputs=predictions,name='vgg')
    #model = Model(inputs=model_input, outputs=x,name='concatanetedmodel')
    return model

from keras.layers.merge import concatenate
import keras
from keras.models import Model, Input
model_input = Input(shape=(96,96,3))

model=buildmodel(model_input)

model.compile(optimizer='Adam', loss='binary_crossentropy',metrics=['accuracy'])
model.summary()

import numpy as np
from keras.utils import np_utils

indices=list(range(len(Images)))
np.random.seed(23)
np.random.shuffle(indices)
Images = np.array(Images)
Images= Images/255.0
labels = np.array(labels)
#labels=np_utils.to_categorical(labels)
ind=int(len(indices)*0.85)
# train data
X_train=Images[indices[:ind]] 
y_train=labels[indices[:ind]]
# validation data
X_val=Images[indices[-(len(indices)-ind):]] 
y_val=labels[indices[-(len(indices)-ind):]]

#image_gen.fit(X_train, augment=False)
from keras.callbacks import ModelCheckpoint
path_model='model_simple_keras_starter.h5' # save model at this location after each epoch
checkpointer = ModelCheckpoint('model.h5', verbose=1, save_best_only=True)
model.fit(x=X_train, y=y_train,
            epochs=40, 
            verbose=1, 
            validation_data=(X_val,y_val),
            shuffle=True,
          callbacks=[
                checkpointer,
            ]
            
            )