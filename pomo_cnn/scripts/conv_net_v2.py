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
from tensorflow.keras.layers import Conv2D,Dropout,Dense,MaxPooling2D
from tensorflow.keras.layers import Input,Flatten,BatchNormalization,Add
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping,ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
def read_pictures(path):
    image_array = plt.imread(path)[:,:,0]
    label = path.split('\\')[-1]
    label = label.split("2")[0]
    return image_array,label

def div_to_float(img):
    image=np.array(img)
    return image/255.

datagen_train = ImageDataGenerator(fill_mode='constant',cval=0,
                             rotation_range=130,
                             width_shift_range=0.4,
                             height_shift_range=0.4,
                             vertical_flip = True,
                             horizontal_flip=True,
                             channel_shift_range=0.3,
                             preprocessing_function = div_to_float)

datagen_val_test = ImageDataGenerator(preprocessing_function = div_to_float)
# load and iterate training dataset
train_it = datagen_train.flow_from_directory(r'C:\Users\mishu\OneDrive\Desktop\projects\Image classifications\conv\data\train\\',color_mode="grayscale",shuffle=True, class_mode='categorical',target_size=(360, 360) ,batch_size=32)
# load and iterate validation dataset
val_it = datagen_val_test.flow_from_directory(r'C:\Users\mishu\OneDrive\Desktop\projects\Image classifications\conv\data\validation\\',color_mode="grayscale",shuffle=True, class_mode='categorical',target_size=(360, 360), batch_size=16)
# load and iterate test dataset
test_it = datagen_val_test.flow_from_directory(r'C:\Users\mishu\OneDrive\Desktop\projects\Image classifications\conv\data\test\\',color_mode="grayscale",shuffle=True, class_mode='categorical',target_size=(360, 360), batch_size=16)



input_image = Input(shape=(360, 360,1))
x = Conv2D(64, (3, 3),strides=1,padding='same')(input_image)
x = Conv2D(64, (3, 3),strides=2,padding='same', activation='relu')(x)
y = Conv2D(64, (1, 1),strides=2,padding='same')(input_image)
x_res = Add()([x, y])
x = MaxPooling2D(pool_size=(2, 2))(x_res)

x = Conv2D(64, (3, 3),strides=1,padding='same')(x)
x = Conv2D(64, (3, 3),strides=2,padding='same', activation='relu')(x)
y = MaxPooling2D(pool_size=(2, 2))(y)
y = Conv2D(64, (1, 1),strides=2,padding='same')(y)
x_res = Add()([x, y])
x = MaxPooling2D(pool_size=(2, 2))(x_res)

x = Conv2D(128, (3, 3),strides=1,padding='same')(x)
x = Conv2D(128, (3, 3),strides=2,padding='same', activation='relu')(x)
y = MaxPooling2D(pool_size=(2, 2))(y)
y = Conv2D(128, (1, 1),strides=2,padding='same')(y)
x_res = Add()([x, y])
x = MaxPooling2D(pool_size=(2, 2))(x_res)




x_flat = Flatten()(x)
x_dense = Dense(512,activation="relu")(x_flat)
x_dense = Dropout(0.25)(x_dense)
x_dense = Dense(256,activation="relu")(x_flat)
x_dense = Dropout(0.25)(x_dense)
x_out = Dense(10,activation="softmax")(x_flat)

model = Model(input_image,x_out)
adam = Adam(learning_rate=0.0005, beta_1=0.8, beta_2=0.999, amsgrad=True) 

model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['categorical_accuracy'])
model.summary()
# checkpoint

filepath="weights-improvement-{epoch:02d}-{val_categorical_accuracy:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath,
                             monitor='val_categorical_accuracy',
                             verbose=1,
                             save_best_only=True,
                             mode='max')
reduce_lr = ReduceLROnPlateau(monitor='val_categorical_accuracy',
                              mode='max',
                              verbose = 1,
                              factor=0.5,
                              patience=10,
                              cooldown=2,
                              min_lr=0.00000001)
es = EarlyStopping(monitor='val_categorical_accuracy',
                   mode='max',
                   verbose=1,
                   patience=50)
callbacks_list = [checkpoint,reduce_lr,es]
#callbacks=callbacks_list
history = model.fit(train_it,
          epochs=100,
          callbacks=callbacks_list,
          verbose=1,
          shuffle=True,
          validation_data=val_it)

plt.figure(figsize=(10,10))  
plt.plot(history.history['categorical_accuracy'],'r')  
plt.plot(history.history['val_categorical_accuracy'],'g')  
#plt.xticks(np.arange(0, 11, 2.0))  
plt.rcParams['figure.figsize'] = (8, 6)  
plt.xlabel("Num of Epochs")  
plt.ylabel("Accuracy")  
plt.title("Training Accuracy vs Validation Accuracy")  
plt.legend(['train','validation'])



plt.figure(figsize=(10,10))  
plt.plot(history.history['loss'],'r')  
plt.plot(history.history['val_loss'],'g')  
#plt.xticks(np.arange(0, 11, 2.0))  
plt.rcParams['figure.figsize'] = (8, 6)  
plt.xlabel("Num of Epochs")  
plt.ylabel("Loss")  
plt.title("Training Loss vs Validation Loss")  
plt.legend(['train','validation'])

plt.show() 


print(model.predict(test_it))
model.save("best_model.h5")