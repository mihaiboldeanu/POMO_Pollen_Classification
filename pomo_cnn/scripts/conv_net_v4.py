# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 12:39:22 2019

@author: mishu
"""
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential,load_model,Model
from tensorflow.keras.layers import Conv2D,Dropout,Dense,MaxPooling2D,Activation,SeparableConv2D 
from tensorflow.keras.layers import Input,Flatten,BatchNormalization,Add,GlobalAveragePooling2D,LeakyReLU
from tensorflow.keras import regularizers
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping,ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam,SGD
from tensorflow.keras.mixed_precision import experimental as mixed_precision
import tensorflow as tf
import json 
import lib_pomo
from tensorflow.keras.applications import ResNet50,Xception
tf.keras.backend.clear_session() 
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
def read_pictures(path):
    image_array = plt.imread(path)[:,:,0]
    label = path.split('\\')[-1]
    label = label.split("2")[0]
    return image_array,label

def div_to_float(img):
    image=np.array(img)
    return image/255.

datagen_train = ImageDataGenerator( rescale=1/255.,fill_mode='constant',cval=0,
                                    rotation_range=360,
                                    channel_shift_range=32,
                                    zoom_range = 0.1,
                                    shear_range = 0.2,
                                    vertical_flip=True,
                                    horizontal_flip=True)

datagen_val_test = ImageDataGenerator(rescale=1/255.,)
# load and iterate training dataset
train_it_aug = datagen_train.flow_from_directory(r'C:\Users\Mihai Boldeanu\Desktop\pollen_clas\data\train\\',
                                             color_mode="grayscale",
                                             shuffle=True,
                                             class_mode='categorical',
                                             target_size=(360, 360) ,
                                             batch_size=64,)
# load and iterate test dataset
train_it = datagen_val_test.flow_from_directory(r'C:\Users\Mihai Boldeanu\Desktop\pollen_clas\data\train\\',
                                               color_mode="grayscale",
                                               shuffle=True,
                                               class_mode='categorical',
                                               target_size=(360, 360),
                                               batch_size=64)


# load and iterate validation dataset
val_it = datagen_val_test.flow_from_directory(r'C:\Users\Mihai Boldeanu\Desktop\pollen_clas\data\validation\\',
                                              color_mode="grayscale",
                                              shuffle=False,
                                              class_mode='categorical',
                                              target_size=(360, 360),
                                              batch_size=64)




model_id = 9
print(model_id)
model = lib_pomo.build_cnn(model_id)
adam = Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, amsgrad=False)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['categorical_accuracy'])
# model.summary()

reduce_lr = ReduceLROnPlateau(monitor='loss',
                              mode='min',
                              verbose = 1,
                              factor=0.9,
                              patience=20,
                              cooldown=2,
                              min_lr=0.00000001)


filepath="cnn_model-"+str(model_id)+"-aug-{val_categorical_accuracy:.5f}.hdf5"
checkpoint_aug = ModelCheckpoint(filepath,
                             monitor='val_categorical_accuracy',
                             verbose=1,
                             save_best_only=True,
                             mode='max')

callbacks_list_aug = [checkpoint_aug,reduce_lr]

history = model.fit(train_it_aug,
                    epochs=500,
                    callbacks=callbacks_list_aug,
                    verbose=1,
                    shuffle=True,
                    validation_data=val_it,
                    workers=4)  
np.save("cnn_model_"+str(model_id)+"_history_aug.npy",history.history)

# model = load_model("cnn_model-6-aug-0.90904.hdf5")

history = model.fit(train_it,
                    epochs=100,
                    callbacks=callbacks_list_aug,
                    verbose=1,
                    shuffle=True,
                    validation_data=val_it,
                    workers=4)

np.save("cnn_model_"+str(model_id)+"_history.npy",history.history)


# ceva = train_it_aug.next()
# image = ceva[0][1]
# plt.figure()
# plt.imshow(image,cmap="gray")
# image = ceva[0][1]>0

# plt.figure()
# plt.imshow(image,cmap="gray")

# plt.figure(figsize=(10,10))  
# plt.plot(history.history['categorical_accuracy'],'r')  
# plt.plot(history.history['val_categorical_accuracy'],'g')  
# #plt.xticks(np.arange(0, 11, 2.0))  
# plt.rcParams['figure.figsize'] = (8, 6)  
# plt.xlabel("Num of Epochs")  
# plt.ylabel("Accuracy")  
# plt.title("Training Accuracy vs Validation Accuracy")  
# plt.legend(['train','validation'])



# plt.figure(figsize=(10,10))  
# plt.plot(history.history['loss'],'r')  
# plt.plot(history.history['val_loss'],'g')  
# #plt.xticks(np.arange(0, 11, 2.0))  
# plt.rcParams['figure.figsize'] = (8, 6)  
# plt.xlabel("Num of Epochs")  
# plt.ylabel("Loss")  
# plt.title("Training Loss vs Validation Loss")  
# plt.legend(['train','validation'])
# plt.show() 

# history.history['categorical_accuracy'],'r')  
# plt.plot(history.history['val_categorical_accuracy'],'g')  
# #plt.xticks(np.arange(0, 11, 2.0))  
# plt.rcParams['figure.figsize'] = (8, 6)  
# plt.xlabel("Num of Epochs")  
# plt.ylabel("Accuracy")  
# plt.title("Training Accuracy vs Validation Accuracy")  
# plt.legend(['train','validation'])



# plt.figure(figsize=(10,10))  
# plt.plot(history.history['loss'],'r')  
# plt.plot(history.history['val_loss'],'g')  

