# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 07:45:38 2021

@author: Mihai Boldeanu
"""
from tensorflow.keras import regularizers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D,Dropout,Dense,MaxPooling2D
from tensorflow.keras.layers import Activation,GlobalAveragePooling2D
from tensorflow.keras.layers import Input,Flatten,BatchNormalization
from tensorflow.keras.layers import Concatenate





def build_cnn(model_id):
    l1_weight = 1e-6
    l2_weight = 1e-5
    # # # ##canonical
    input_image = Input(shape=(360, 360,1))
    ###### Multiple configureations may the best one win
    if model_id==0:
        x = Conv2D(16, (3, 3),strides=1,padding='same',use_bias=False,kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(input_image)
        x = Conv2D(16, (3, 3),strides=1,padding='same',use_bias=False,kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x)
        x = Conv2D(16, (3, 3),strides=1,padding='same',use_bias=False,kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Activation("relu")(x)
        x = BatchNormalization()(x) 
        
        
        x = Conv2D(32, (3, 3),strides=2,padding='same',use_bias=False,kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x)
        x = Conv2D(32, (3, 3),strides=1,padding='same',use_bias=False,kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x)
        x = Conv2D(32, (3, 3),strides=1,padding='same',use_bias=False,kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Activation("relu")(x)
        x = BatchNormalization()(x) 
        
        
        x = Conv2D(64, (3, 3),strides=2,padding='same',use_bias=False,kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x)
        x = Conv2D(64, (3, 3),strides=1,padding='same',use_bias=False,kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x)
        x = Conv2D(64, (3, 3),strides=1,padding='same',use_bias=False,kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Activation("relu")(x)
        x = BatchNormalization()(x) 
        
        
        
        x = Conv2D(128, (3, 3),strides=2,padding='same',use_bias=False,kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x)
        x = Conv2D(128, (3, 3),strides=1,padding='same',use_bias=False,kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x)
        x = Activation("relu")(x)
        x = BatchNormalization()(x) 


        x_flat = Flatten()(x)
        x_dense = Dense(256,activation="relu",kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x_flat)
        x_dense = Dropout(0.25)(x_dense)
        x_dense = Dense(128,activation="relu",kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x_dense)
        x_dense = Dropout(0.25)(x_dense)
        x_dense = Dense(64,activation="relu",kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x_dense)
        x_dense = Dropout(0.25)(x_dense)
        x_out = Dense(19)(x_dense)
        x_out = Activation('softmax',  name='predictions')(x_out)
        
        
        model = Model(input_image,x_out)
        return model
    if model_id==1:
        x = Conv2D(64, (3, 3),strides=1,padding='same',use_bias=False,kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(input_image)
        x = Conv2D(64, (3, 3),strides=1,padding='same',use_bias=False,kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x)
        x = Conv2D(64, (3, 3),strides=1,padding='same',use_bias=False,kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Activation("relu")(x)
        x = BatchNormalization()(x) 
        
        
        x = Conv2D(64, (3, 3),strides=2,padding='same',use_bias=False,kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x)
        x = Conv2D(64, (3, 3),strides=1,padding='same',use_bias=False,kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x)
        x = Conv2D(64, (3, 3),strides=1,padding='same',use_bias=False,kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Activation("relu")(x)
        x = BatchNormalization()(x) 
        
        
        x = Conv2D(128, (3, 3),strides=2,padding='same',use_bias=False,kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x)
        x = Conv2D(128, (3, 3),strides=1,padding='same',use_bias=False,kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x)
        x = Conv2D(128, (3, 3),strides=1,padding='same',use_bias=False,kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Activation("relu")(x)
        x = BatchNormalization()(x) 
        
        
        
        x = Conv2D(256, (3, 3),strides=2,padding='same',use_bias=False,kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x)
        x = Conv2D(256, (3, 3),strides=1,padding='same',use_bias=False,kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x)
        x = Activation("relu")(x)
        x = BatchNormalization()(x) 
        
        
        
        x = Conv2D(512, (3, 3),strides=2,padding='same',use_bias=False,kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x)
        x = Activation("relu")(x)
        x = BatchNormalization()(x) 
        
        
        
        # x_dense = GlobalAveragePooling2D()(x)
        x_flat = Flatten()(x)
        x_dense = Dense(256,activation="relu",kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x_flat)
        x_dense = Dropout(0.25)(x_dense)
        x_dense = Dense(128,activation="relu",kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x_dense)
        x_dense = Dropout(0.25)(x_dense)
        x_dense = Dense(64,activation="relu",kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x_dense)
        x_dense = Dropout(0.25)(x_dense)
        x_out = Dense(18)(x_dense)
        x_out = Activation('softmax',  name='predictions')(x_out)
        
        
        model = Model(input_image,x_out)
        return model
    if model_id==2:
        x = Conv2D(64, (3, 3),strides=1,padding='same',use_bias=False,kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(input_image)
        x = Conv2D(64, (3, 3),strides=1,padding='same',use_bias=False,kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x)
        x = Conv2D(64, (3, 3),strides=1,padding='same',use_bias=False,kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Activation("relu")(x)
        x = BatchNormalization()(x) 
        
        
        x = Conv2D(64, (3, 3),strides=2,padding='same',use_bias=False,kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x)
        x = Conv2D(64, (3, 3),strides=1,padding='same',use_bias=False,kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x)
        x = Conv2D(64, (3, 3),strides=1,padding='same',use_bias=False,kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Activation("relu")(x)
        x = BatchNormalization()(x) 
        
        
        x = Conv2D(128, (3, 3),strides=2,padding='same',use_bias=False,kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x)
        x = Conv2D(128, (3, 3),strides=1,padding='same',use_bias=False,kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x)
        x = Conv2D(128, (3, 3),strides=1,padding='same',use_bias=False,kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Activation("relu")(x)
        x = BatchNormalization()(x) 
        
        
        
        x = Conv2D(256, (3, 3),strides=2,padding='same',use_bias=False,kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x)
        x = Conv2D(256, (3, 3),strides=1,padding='same',use_bias=False,kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x)
        x = Activation("relu")(x)
        x = BatchNormalization()(x) 
        
        
        
        x = Conv2D(512, (3, 3),strides=2,padding='same',use_bias=False,kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x)
        x = Activation("relu")(x)
        x = BatchNormalization()(x) 
        
        
        
        x_dense = GlobalAveragePooling2D()(x)
        
        x_out = Dense(18)(x_dense)
        x_out = Activation('softmax',  name='predictions')(x_out)
        
        
        model = Model(input_image,x_out)
        return model
    if model_id==3:
        x = Conv2D(64, (3, 3),strides=1,padding='same',use_bias=False,kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(input_image)
        x = Conv2D(64, (3, 3),strides=1,padding='same',use_bias=False,kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x)
        x = Conv2D(64, (3, 3),strides=1,padding='same',use_bias=False,kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x)
        x = Conv2D(64, (3, 3),strides=1,padding='same',use_bias=False,kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x)
        x = Conv2D(64, (3, 3),strides=1,padding='same',use_bias=False,kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x)
        x = Conv2D(64, (3, 3),strides=1,padding='same',use_bias=False,kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Activation("relu")(x)
        x = BatchNormalization()(x) 
        
        
        x = Conv2D(64, (3, 3),strides=2,padding='same',use_bias=False,kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x)
        x = Conv2D(64, (3, 3),strides=1,padding='same',use_bias=False,kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x)
        x = Conv2D(64, (3, 3),strides=1,padding='same',use_bias=False,kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x)
        x = Conv2D(64, (3, 3),strides=1,padding='same',use_bias=False,kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x)
        x = Conv2D(64, (3, 3),strides=1,padding='same',use_bias=False,kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Activation("relu")(x)
        x = BatchNormalization()(x) 
        
        
        x = Conv2D(128, (3, 3),strides=2,padding='same',use_bias=False,kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x)
        x = Conv2D(128, (3, 3),strides=1,padding='same',use_bias=False,kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x)
        x = Conv2D(128, (3, 3),strides=1,padding='same',use_bias=False,kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Activation("relu")(x)
        x = BatchNormalization()(x) 
        
        
        
        x = Conv2D(256, (3, 3),strides=2,padding='same',use_bias=False,kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x)
        x = Conv2D(256, (3, 3),strides=1,padding='same',use_bias=False,kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x)
        x = Activation("relu")(x)
        x = BatchNormalization()(x) 
        
        
        
        x = Conv2D(512, (3, 3),strides=2,padding='same',use_bias=False,kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x)
        x = Activation("relu")(x)
        x = BatchNormalization()(x) 
        
        
        
        # x_dense = GlobalAveragePooling2D()(x)
        x_flat = Flatten()(x)
        x_dense = Dense(256,activation="relu",kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x_flat)
        x_dense = Dropout(0.25)(x_dense)
        x_dense = Dense(128,activation="relu",kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x_dense)
        x_dense = Dropout(0.25)(x_dense)
        x_dense = Dense(64,activation="relu",kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x_dense)
        x_dense = Dropout(0.25)(x_dense)
        x_out = Dense(18)(x_dense)
        x_out = Activation('softmax',  name='predictions')(x_out)
        
        
        model = Model(input_image,x_out)
        return model
    if model_id==4:
        x = Conv2D(64, (3, 3),strides=1,padding='same',use_bias=False,kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(input_image)
        x = Conv2D(64, (3, 3),strides=1,padding='same',use_bias=False,kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x)
        x = Conv2D(64, (3, 3),strides=1,padding='same',use_bias=False,kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Activation("relu")(x)
        x = BatchNormalization()(x) 
        
        
        x = Conv2D(64, (3, 3),strides=2,padding='same',use_bias=False,kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x)
        x = Conv2D(64, (3, 3),strides=1,padding='same',use_bias=False,kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x)
        x = Conv2D(64, (3, 3),strides=1,padding='same',use_bias=False,kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Activation("relu")(x)
        x = BatchNormalization()(x) 
        
        
        x = Conv2D(128, (3, 3),strides=2,padding='same',use_bias=False,kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x)
        x = Conv2D(128, (3, 3),strides=1,padding='same',use_bias=False,kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x)
        x = Conv2D(128, (3, 3),strides=1,padding='same',use_bias=False,kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Activation("relu")(x)
        x = BatchNormalization()(x) 
        
        
        
        x = Conv2D(256, (3, 3),strides=2,padding='same',use_bias=False,kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x)
        x = Conv2D(256, (3, 3),strides=1,padding='same',use_bias=False,kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x)
        x = Activation("relu")(x)
        x = BatchNormalization()(x) 
        
        
        
        x = Conv2D(512, (3, 3),strides=2,padding='same',use_bias=False,kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x)
        x = Activation("relu")(x)
        x = BatchNormalization()(x) 
        
        
        
        # x_dense = GlobalAveragePooling2D()(x)
        x_flat = Flatten()(x)
        x_dense = Dense(256,activation="relu",kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x_flat)
        x_dense = Dropout(0.25)(x_dense)
        x_dense = Dense(256,activation="relu",kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x_dense)
        x_dense = Dropout(0.25)(x_dense)
        x_dense = Dense(256,activation="relu",kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x_dense)
        x_dense = Dropout(0.25)(x_dense)
        x_out = Dense(18)(x_dense)
        x_out = Activation('softmax',  name='predictions')(x_out)
        
        
        model = Model(input_image,x_out)
        return model
    if model_id==5:
        x = Conv2D(64, (3, 3),strides=1,padding='same',use_bias=False,kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(input_image)
        x = Conv2D(64, (3, 3),strides=1,padding='same',use_bias=False,kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x)
        x = Conv2D(64, (3, 3),strides=1,padding='same',use_bias=False,kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Activation("relu")(x)
        x = BatchNormalization()(x) 
        
        
        x = Conv2D(64, (3, 3),strides=2,padding='same',use_bias=False,kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x)
        x = Conv2D(64, (3, 3),strides=1,padding='same',use_bias=False,kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x)
        x = Conv2D(64, (3, 3),strides=1,padding='same',use_bias=False,kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Activation("relu")(x)
        x = BatchNormalization()(x) 
        
        
        x = Conv2D(128, (3, 3),strides=2,padding='same',use_bias=False,kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x)
        x = Conv2D(128, (3, 3),strides=1,padding='same',use_bias=False,kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x)
        x = Conv2D(128, (3, 3),strides=1,padding='same',use_bias=False,kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Activation("relu")(x)
        x = BatchNormalization()(x) 
        
        
        
        x = Conv2D(256, (3, 3),strides=2,padding='same',use_bias=False,kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x)
        x = Conv2D(256, (3, 3),strides=1,padding='same',use_bias=False,kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x)
        x = Activation("relu")(x)
        x = BatchNormalization()(x) 
        
        
        
        x = Conv2D(512, (3, 3),strides=2,padding='same',use_bias=False,kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x)
        x = Activation("relu")(x)
        x = BatchNormalization()(x) 
        
        
        
        # x_dense = GlobalAveragePooling2D()(x)
        x_flat = Flatten()(x)
        x_dense = Dense(256,activation="relu",kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x_flat)
        x_dense = Dropout(0.25)(x_dense)
        x_dense = Dense(256,activation="relu",kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x_dense)
        x_dense = Dropout(0.25)(x_dense)
        x_dense = Dense(256,activation="relu",kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x_dense)
        x_dense = Dropout(0.25)(x_dense)
        x_dense = Dense(256,activation="relu",kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x_dense)
        x_dense = Dropout(0.25)(x_dense)
        x_dense = Dense(256,activation="relu",kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x_dense)
        x_dense = Dropout(0.25)(x_dense)
        x_out = Dense(18)(x_dense)
        x_out = Activation('softmax',  name='predictions')(x_out)
        
        
        model = Model(input_image,x_out)
        return model
    if model_id==6:
        x = Conv2D(16, (3, 3),strides=1,padding='same',use_bias=False,kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(input_image)
        x = Conv2D(16, (3, 3),strides=1,padding='same',use_bias=False,kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x)
        x = Conv2D(16, (3, 3),strides=1,padding='same',use_bias=False,kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x)
        x = Conv2D(16, (3, 3),strides=1,padding='same',use_bias=False,kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x)
        x = Conv2D(16, (3, 3),strides=1,padding='same',use_bias=False,kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x)
        x = Conv2D(16, (3, 3),strides=1,padding='same',use_bias=False,kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x)
        x = Conv2D(16, (3, 3),strides=1,padding='same',use_bias=False,kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x)
        x = Conv2D(16, (3, 3),strides=1,padding='same',use_bias=False,kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x)
        x = Conv2D(16, (3, 3),strides=1,padding='same',use_bias=False,kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Activation("relu")(x)
        x = BatchNormalization()(x) 
        
        
        x = Conv2D(32, (3, 3),strides=2,padding='same',use_bias=False,kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x)
        x = Conv2D(32, (3, 3),strides=1,padding='same',use_bias=False,kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x)
        x = Conv2D(32, (3, 3),strides=1,padding='same',use_bias=False,kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x)
        x = Conv2D(32, (3, 3),strides=1,padding='same',use_bias=False,kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x)
        x = Conv2D(32, (3, 3),strides=1,padding='same',use_bias=False,kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Activation("relu")(x)
        x = BatchNormalization()(x) 
        
        
        x = Conv2D(64, (3, 3),strides=2,padding='same',use_bias=False,kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x)
        x = Conv2D(64, (3, 3),strides=1,padding='same',use_bias=False,kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x)
        x = Conv2D(64, (3, 3),strides=1,padding='same',use_bias=False,kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x)
        x = Conv2D(64, (3, 3),strides=1,padding='same',use_bias=False,kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x)
        x = Conv2D(64, (3, 3),strides=1,padding='same',use_bias=False,kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Activation("relu")(x)
        x = BatchNormalization()(x) 
        
        
        
        x = Conv2D(128, (3, 3),strides=2,padding='same',use_bias=False,kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x)
        x = Conv2D(128, (3, 3),strides=1,padding='same',use_bias=False,kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x)
        x = Activation("relu")(x)
        x = BatchNormalization()(x) 
        
        
        
        x = Conv2D(256, (3, 3),strides=2,padding='same',use_bias=False,kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x)
        x = Activation("relu")(x)
        x = BatchNormalization()(x) 
        
        
        
        # x_dense = GlobalAveragePooling2D()(x)
        x_flat = Flatten()(x)
        x_dense = Dense(256,activation="relu",kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x_flat)
        x_dense = Dropout(0.25)(x_dense)
        x_dense = Dense(128,activation="relu",kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x_dense)
        x_dense = Dropout(0.25)(x_dense)
        x_dense = Dense(64,activation="relu",kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x_dense)
        x_dense = Dropout(0.25)(x_dense)
        x_out = Dense(18)(x_dense)
        x_out = Activation('softmax',  name='predictions')(x_out)
        
        
        model = Model(input_image,x_out)
        return model
    if model_id==7:
        x = Conv2D(64, (3, 3),strides=1,padding='same',use_bias=False,kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(input_image)
        x = Conv2D(64, (3, 3),strides=1,padding='same',use_bias=False,kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x)
        x = Conv2D(64, (3, 3),strides=1,padding='same',use_bias=False,kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Activation("relu")(x)
        x = BatchNormalization()(x) 
        
        
        x = Conv2D(128, (3, 3),strides=2,padding='same',use_bias=False,kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x)
        x = Conv2D(64, (3, 3),strides=1,padding='same',use_bias=False,kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x)
        x = Conv2D(64, (3, 3),strides=1,padding='same',use_bias=False,kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Activation("relu")(x)
        x = BatchNormalization()(x) 
        
        
        x = Conv2D(256, (3, 3),strides=2,padding='same',use_bias=False,kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x)
        x = Conv2D(128, (3, 3),strides=1,padding='same',use_bias=False,kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x)
        x = Conv2D(128, (3, 3),strides=1,padding='same',use_bias=False,kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Activation("relu")(x)
        x = BatchNormalization()(x) 
        
        
        
        x = Conv2D(512, (3, 3),strides=2,padding='same',use_bias=False,kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x)
        x = Conv2D(256, (3, 3),strides=1,padding='same',use_bias=False,kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x)
        x = Activation("relu")(x)
        x = BatchNormalization()(x) 
        
        
        
        x = Conv2D(512, (3, 3),strides=2,padding='same',use_bias=False,kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x)
        x = Activation("relu")(x)
        x = BatchNormalization()(x) 
        
        
        
        # x_dense = GlobalAveragePooling2D()(x)
        x_flat = Flatten()(x)
        x_dense = Dense(256,activation="relu",kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x_flat)
        x_dense = Dropout(0.25)(x_dense)
        x_dense = Dense(128,activation="relu",kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x_dense)
        x_dense = Dropout(0.25)(x_dense)
        x_dense = Dense(64,activation="relu",kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x_dense)
        x_dense = Dropout(0.25)(x_dense)
        x_out = Dense(18)(x_dense)
        x_out = Activation('softmax',  name='predictions')(x_out)
        
        
        model = Model(input_image,x_out)
        return model
    if model_id==8:
        x = Conv2D(10, (3, 3),strides=1,padding='same',use_bias=False,kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(input_image)
        x = Conv2D(10, (3, 3),strides=1,padding='same',use_bias=False,kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x)
        x = Conv2D(10, (3, 3),strides=1,padding='same',use_bias=False,kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Activation("relu")(x)
        x = BatchNormalization()(x) 
        
        
        x = Conv2D(20, (3, 3),strides=2,padding='same',use_bias=False,kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x)
        x = Conv2D(20, (3, 3),strides=1,padding='same',use_bias=False,kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x)
        x = Conv2D(20, (3, 3),strides=1,padding='same',use_bias=False,kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Activation("relu")(x)
        x = BatchNormalization()(x) 
        
        
        x = Conv2D(30, (3, 3),strides=2,padding='same',use_bias=False,kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x)
        x = Conv2D(30, (3, 3),strides=1,padding='same',use_bias=False,kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x)
        x = Conv2D(30, (3, 3),strides=1,padding='same',use_bias=False,kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Activation("relu")(x)
        x = BatchNormalization()(x) 
        
        
        
        x = Conv2D(50, (3, 3),strides=2,padding='same',use_bias=False,kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x)
        x = Conv2D(50, (3, 3),strides=1,padding='same',use_bias=False,kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x)
        x = Activation("relu")(x)
        x = BatchNormalization()(x) 
        
        
        
        x = Conv2D(70, (3, 3),strides=2,padding='same',use_bias=False,kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x)
        x = Activation("relu")(x)
        x = BatchNormalization()(x) 
        
        
        
        # x_dense = GlobalAveragePooling2D()(x)
        x_flat = Flatten()(x)
        x_dense = Dense(256,activation="relu",kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x_flat)
        x_dense = Dropout(0.25)(x_dense)
        x_dense = Dense(128,activation="relu",kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x_dense)
        x_dense = Dropout(0.25)(x_dense)
        x_dense = Dense(64,activation="relu",kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x_dense)
        x_dense = Dropout(0.25)(x_dense)
        x_out = Dense(18)(x_dense)
        x_out = Activation('softmax',  name='predictions')(x_out)
        
        
        model = Model(input_image,x_out)
        return model
    if model_id==9:
        x = Conv2D(64, (3, 3),strides=1,padding='same',use_bias=False,kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(input_image)
        x = Conv2D(64, (3, 3),strides=1,padding='same',use_bias=False,kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x)
        x = Conv2D(64, (3, 3),strides=1,padding='same',use_bias=False,kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Activation("relu")(x)
        x = BatchNormalization()(x) 
        
        
        x = Conv2D(64, (3, 3),strides=2,padding='same',use_bias=False,kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x)
        x = Conv2D(64, (3, 3),strides=1,padding='same',use_bias=False,kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x)
        x = Conv2D(64, (3, 3),strides=1,padding='same',use_bias=False,kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Activation("relu")(x)
        x = BatchNormalization()(x) 
        
        
        x = Conv2D(128, (3, 3),strides=2,padding='same',use_bias=False,kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x)
        x = Conv2D(128, (3, 3),strides=1,padding='same',use_bias=False,kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x)
        x = Conv2D(128, (3, 3),strides=1,padding='same',use_bias=False,kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Activation("relu")(x)
        x = BatchNormalization()(x) 
        
        
        
        x = Conv2D(256, (3, 3),strides=2,padding='same',use_bias=False,kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x)
        x = Activation("relu")(x)
        x = BatchNormalization()(x) 

        
        
        # x_dense = GlobalAveragePooling2D()(x)
        x_flat = Flatten()(x)
        x_dense = Dense(256,activation="relu",kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x_flat)
        x_dense = Dropout(0.25)(x_dense)
        x_dense = Dense(128,activation="relu",kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x_dense)
        x_dense = Dropout(0.25)(x_dense)
        x_dense = Dense(64,activation="relu",kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x_dense)
        x_dense = Dropout(0.25)(x_dense)
        x_out = Dense(18)(x_dense)
        x_out = Activation('softmax',  name='predictions')(x_out)
        
        
        model = Model(input_image,x_out)
        return model
    if model_id==10:
        x = Conv2D(64, (3, 3),strides=1,padding='same',use_bias=False,kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(input_image)
        x = Conv2D(64, (3, 3),strides=1,padding='same',use_bias=False,kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x)
        x = Conv2D(64, (3, 3),strides=1,padding='same',use_bias=False,kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Activation("relu")(x)
        x = BatchNormalization()(x) 
        
        
        x = Conv2D(128, (3, 3),strides=2,padding='same',use_bias=False,kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x)
        x = Conv2D(128, (3, 3),strides=1,padding='same',use_bias=False,kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x)
        x = Conv2D(128, (3, 3),strides=1,padding='same',use_bias=False,kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Activation("relu")(x)
        x = BatchNormalization()(x) 
        
        
        x = Conv2D(256, (3, 3),strides=2,padding='same',use_bias=False,kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x)
        x = Conv2D(256, (3, 3),strides=1,padding='same',use_bias=False,kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x)
        x = Conv2D(256, (3, 3),strides=1,padding='same',use_bias=False,kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Activation("relu")(x)
        x = BatchNormalization()(x) 
        
        
        x_dense = GlobalAveragePooling2D()(x)
        x_out = Dense(18)(x_dense)
        x_out = Activation('softmax',  name='predictions')(x_out)
        
        
        model = Model(input_image,x_out)
        return model
    if model_id==11:
        x = Conv2D(64, (3, 3),strides=1,padding='same',use_bias=False,kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(input_image)
        x = Conv2D(64, (3, 3),strides=1,padding='same',use_bias=False,kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x)
        x = Conv2D(64, (3, 3),strides=1,padding='same',use_bias=False,kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Activation("relu")(x)
        x = BatchNormalization()(x) 
        
        
        x = Conv2D(64, (3, 3),strides=2,padding='same',use_bias=False,kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x)
        x = Conv2D(64, (3, 3),strides=1,padding='same',use_bias=False,kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x)
        x = Conv2D(64, (3, 3),strides=1,padding='same',use_bias=False,kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Activation("relu")(x)
        x = BatchNormalization()(x) 
        
        
        x = Conv2D(128, (3, 3),strides=2,padding='same',use_bias=False,kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x)
        x = Conv2D(128, (3, 3),strides=1,padding='same',use_bias=False,kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x)
        x = Conv2D(128, (3, 3),strides=1,padding='same',use_bias=False,kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Activation("relu")(x)
        x = BatchNormalization()(x) 
        
        
        
        x = Conv2D(256, (3, 3),strides=2,padding='same',use_bias=False,kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x)
        x = Conv2D(256, (3, 3),strides=1,padding='same',use_bias=False,kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x)
        x = Activation("relu")(x)
        x = BatchNormalization()(x) 
        
        
        
        x = Conv2D(512, (3, 3),strides=2,padding='same',use_bias=False,kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x)
        x = Activation("relu")(x)
        x = BatchNormalization()(x) 
        
        
        
        # x_dense = GlobalAveragePooling2D()(x)
        x_flat = Flatten()(x)
        x_dense = Dense(256,activation="relu",kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x_flat)
        x_dense = Dropout(0.5)(x_dense)
        x_dense = Dense(128,activation="relu",kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x_dense)
        x_dense = Dropout(0.5)(x_dense)
        x_dense = Dense(64,activation="relu",kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x_dense)
        x_dense = Dropout(0.5)(x_dense)
        x_out = Dense(18)(x_dense)
        x_out = Activation('softmax',  name='predictions')(x_out)
        
        
        model = Model(input_image,x_out)
        return model
    if model_id==12:
        l1_weight = 1e-5
        l2_weight = 1e-4
        x = Conv2D(64, (3, 3),strides=1,padding='same',use_bias=False,kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(input_image)
        x = Conv2D(64, (3, 3),strides=1,padding='same',use_bias=False,kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x)
        x = Conv2D(64, (3, 3),strides=1,padding='same',use_bias=False,kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Activation("relu")(x)
        x = BatchNormalization()(x) 
        
        
        x = Conv2D(64, (3, 3),strides=2,padding='same',use_bias=False,kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x)
        x = Conv2D(64, (3, 3),strides=1,padding='same',use_bias=False,kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x)
        x = Conv2D(64, (3, 3),strides=1,padding='same',use_bias=False,kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Activation("relu")(x)
        x = BatchNormalization()(x) 
        
        
        x = Conv2D(128, (3, 3),strides=2,padding='same',use_bias=False,kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x)
        x = Conv2D(128, (3, 3),strides=1,padding='same',use_bias=False,kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x)
        x = Conv2D(128, (3, 3),strides=1,padding='same',use_bias=False,kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Activation("relu")(x)
        x = BatchNormalization()(x) 
        
        
        
        x = Conv2D(256, (3, 3),strides=2,padding='same',use_bias=False,kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x)
        x = Conv2D(256, (3, 3),strides=1,padding='same',use_bias=False,kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x)
        x = Activation("relu")(x)
        x = BatchNormalization()(x) 
        
        
        
        x = Conv2D(512, (3, 3),strides=2,padding='same',use_bias=False,kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x)
        x = Activation("relu")(x)
        x = BatchNormalization()(x) 
        
        
        
        # x_dense = GlobalAveragePooling2D()(x)
        x_flat = Flatten()(x)
        x_dense = Dense(256,activation="relu",kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x_flat)
        x_dense = Dropout(0.25)(x_dense)
        x_dense = Dense(128,activation="relu",kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x_dense)
        x_dense = Dropout(0.25)(x_dense)
        x_dense = Dense(64,activation="relu",kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x_dense)
        x_dense = Dropout(0.25)(x_dense)
        x_out = Dense(18)(x_dense)
        x_out = Activation('softmax',  name='predictions')(x_out)
        
        
        model = Model(input_image,x_out)
        return model
    if model_id==13:
        x = Conv2D(64, (3, 3),strides=1,padding='same',use_bias=False,kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(input_image)
        x = Conv2D(64, (3, 3),strides=1,padding='same',use_bias=False,kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x)
        x = Conv2D(64, (3, 3),strides=1,padding='same',use_bias=False,kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Activation("relu")(x)
        x = BatchNormalization()(x) 
        
        
        x = Conv2D(64, (3, 3),strides=2,padding='same',use_bias=False,kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x)
        x = Conv2D(64, (3, 3),strides=1,padding='same',use_bias=False,kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x)
        x = Conv2D(64, (3, 3),strides=1,padding='same',use_bias=False,kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Activation("relu")(x)
        x = BatchNormalization()(x) 
        
        
        x = Conv2D(128, (3, 3),strides=2,padding='same',use_bias=False,kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x)
        x = Conv2D(128, (3, 3),strides=1,padding='same',use_bias=False,kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x)
        x = Conv2D(128, (3, 3),strides=1,padding='same',use_bias=False,kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Activation("relu")(x)
        x = BatchNormalization()(x) 
        
        
        
        x = Conv2D(256, (3, 3),strides=2,padding='same',use_bias=False,kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x)
        x = Conv2D(256, (3, 3),strides=1,padding='same',use_bias=False,kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x)
        x = Activation("relu")(x)
        x = BatchNormalization()(x) 
        
        
        
        x = Conv2D(512, (3, 3),strides=2,padding='same',use_bias=False,kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x)
        x = Activation("relu")(x)
        x = BatchNormalization()(x) 
        
        
        
        # x_dense = GlobalAveragePooling2D()(x)
        x_flat = Flatten()(x)
        x_dense = Dense(256,activation="relu",kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x_flat)
        x_dense = Dropout(0.5)(x_dense)
        x_dense = Dense(256,activation="relu",kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x_dense)
        x_dense = Dropout(0.5)(x_dense)
        x_dense = Dense(256,activation="relu",kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x_dense)
        x_dense = Dropout(0.5)(x_dense)
        x_out = Dense(18)(x_dense)
        x_out = Activation('softmax',  name='predictions')(x_out)
        
        
        model = Model(input_image,x_out)
        return model
    if model_id==14:
        x = Conv2D(64, (3, 3),strides=1,padding='same',use_bias=False,kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(input_image)
        x = Conv2D(64, (3, 3),strides=1,padding='same',use_bias=False,kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x)
        x = Conv2D(64, (3, 3),strides=1,padding='same',use_bias=False,kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Activation("relu")(x)
        x = BatchNormalization()(x) 
        
        
        x = Conv2D(64, (3, 3),strides=2,padding='same',use_bias=False,kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x)
        x = Conv2D(64, (3, 3),strides=1,padding='same',use_bias=False,kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x)
        x = Conv2D(64, (3, 3),strides=1,padding='same',use_bias=False,kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Activation("relu")(x)
        x = BatchNormalization()(x) 
        
        
        x = Conv2D(128, (3, 3),strides=2,padding='same',use_bias=False,kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x)
        x = Conv2D(128, (3, 3),strides=1,padding='same',use_bias=False,kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x)
        x = Conv2D(128, (3, 3),strides=1,padding='same',use_bias=False,kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Activation("relu")(x)
        x = BatchNormalization()(x) 
        
        
        
        x = Conv2D(256, (3, 3),strides=2,padding='same',use_bias=False,kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x)
        x = Conv2D(256, (3, 3),strides=1,padding='same',use_bias=False,kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x)
        x = Activation("relu")(x)
        x = BatchNormalization()(x) 
        
        
        
        x = Conv2D(512, (3, 3),strides=2,padding='same',use_bias=False,kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x)
        x = Activation("relu")(x)
        x = BatchNormalization()(x) 
        
        
        
        # x_dense = GlobalAveragePooling2D()(x)
        x_flat = Flatten()(x)
        x_dense = Dense(512,activation="relu",kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x_flat)
        x_dense = Dropout(0.5)(x_dense)
        x_dense = Dense(512,activation="relu",kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x_dense)
        x_dense = Dropout(0.5)(x_dense)
        x_dense = Dense(512,activation="relu",kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x_dense)
        x_dense = Dropout(0.5)(x_dense)
        x_out = Dense(18)(x_dense)
        x_out = Activation('softmax',  name='predictions')(x_out)
        
        
        model = Model(input_image,x_out)
        return model
    if model_id==15:
        x = Conv2D(64, (3, 3),strides=1,padding='same',use_bias=False,kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(input_image)
        x = Conv2D(64, (3, 3),strides=1,padding='same',use_bias=False,kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x)
        x = Conv2D(64, (3, 3),strides=1,padding='same',use_bias=False,kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Activation("relu")(x)
        x = BatchNormalization()(x) 
        
        
        x = Conv2D(64, (3, 3),strides=2,padding='same',use_bias=False,kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x)
        x = Conv2D(64, (3, 3),strides=1,padding='same',use_bias=False,kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x)
        x = Conv2D(64, (3, 3),strides=1,padding='same',use_bias=False,kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Activation("relu")(x)
        x = BatchNormalization()(x) 
        
        
        x = Conv2D(128, (3, 3),strides=2,padding='same',use_bias=False,kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x)
        x = Conv2D(128, (3, 3),strides=1,padding='same',use_bias=False,kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x)
        x = Conv2D(128, (3, 3),strides=1,padding='same',use_bias=False,kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Activation("relu")(x)
        x = BatchNormalization()(x) 
        
        
        
        x = Conv2D(256, (3, 3),strides=2,padding='same',use_bias=False,kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x)
        x = Conv2D(256, (3, 3),strides=1,padding='same',use_bias=False,kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x)
        x = Activation("relu")(x)
        x = BatchNormalization()(x) 

        
        # x_dense = GlobalAveragePooling2D()(x)
        x_flat = Flatten()(x)
        x_dense = Dense(256,activation="relu",kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x_flat)
        x_dense = Dropout(0.5)(x_dense)
        x_dense = Dense(256,activation="relu",kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x_dense)
        x_dense = Dropout(0.5)(x_dense)
        x_dense = Dense(256,activation="relu",kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x_dense)
        x_dense = Dropout(0.5)(x_dense)
        x_out = Dense(18)(x_dense)
        x_out = Activation('softmax',  name='predictions')(x_out)
        
        
        model = Model(input_image,x_out)
        return model
    if model_id==16:
        x = Conv2D(64, (3, 3),strides=1,padding='same',use_bias=False,kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(input_image)
        x = Conv2D(64, (3, 3),strides=1,padding='same',use_bias=False,kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x)
        x = Conv2D(64, (3, 3),strides=1,padding='same',use_bias=False,kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Activation("relu")(x)
        x = BatchNormalization()(x) 
        
        
        x = Conv2D(64, (3, 3),strides=2,padding='same',use_bias=False,kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x)
        x = Conv2D(64, (3, 3),strides=1,padding='same',use_bias=False,kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x)
        x = Conv2D(64, (3, 3),strides=1,padding='same',use_bias=False,kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Activation("relu")(x)
        x = BatchNormalization()(x) 
        
        
        x = Conv2D(64, (3, 3),strides=2,padding='same',use_bias=False,kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x)
        x = Conv2D(64, (3, 3),strides=1,padding='same',use_bias=False,kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x)
        x = Conv2D(64, (3, 3),strides=1,padding='same',use_bias=False,kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Activation("relu")(x)
        x = BatchNormalization()(x) 
        
        
        
        x = Conv2D(64, (3, 3),strides=2,padding='same',use_bias=False,kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x)
        x = Conv2D(64, (3, 3),strides=1,padding='same',use_bias=False,kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x)
        x = Activation("relu")(x)
        x = BatchNormalization()(x) 
    
        
        
        # x_dense = GlobalAveragePooling2D()(x)
        x_flat = Flatten()(x)
        x_dense = Dense(256,activation="relu",kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x_flat)
        x_dense = Dropout(0.5)(x_dense)
        x_dense = Dense(128,activation="relu",kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x_dense)
        x_dense = Dropout(0.5)(x_dense)
        x_dense = Dense(64,activation="relu",kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x_dense)
        x_dense = Dropout(0.5)(x_dense)
        x_out = Dense(18)(x_dense)
        x_out = Activation('softmax',  name='predictions')(x_out)
        
        
        model = Model(input_image,x_out)
        return model
    if model_id==17:
        l1_weight = 1e-6
        l2_weight = 1e-5
        # # # ##canonical
        input_image = Input(shape=(360, 360,1))
        ###### Multiple configureations may the best one win
        
        x = Conv2D(16, (3, 3),strides=1,padding='same',use_bias=False,kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(input_image)
        x = Conv2D(16, (3, 3),strides=1,padding='same',use_bias=False,kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x)
        x = Conv2D(16, (3, 3),strides=1,padding='same',use_bias=False,kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Activation("relu")(x)
        x = BatchNormalization()(x) 
        
        
        x_1 = Conv2D(16, (1, 1),strides=1,padding='same',use_bias=False,kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x)
        x_3 = Conv2D(16, (1, 3),strides=1,padding='same',use_bias=False,kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x)
        x_3 = Conv2D(16, (3, 1),strides=1,padding='same',use_bias=False,kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x_3)
        x_5 = Conv2D(16, (1, 5),strides=1,padding='same',use_bias=False,kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x)
        x_5 = Conv2D(16, (5, 1),strides=1,padding='same',use_bias=False,kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x_5)
        x_7 = Conv2D(16, (1, 7),strides=1,padding='same',use_bias=False,kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x)
        x_7 = Conv2D(16, (7, 1),strides=1,padding='same',use_bias=False,kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x_7)
        x = Concatenate(axis=-1)([x_1,x_3,x_5,x_7])
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Activation("relu")(x)
        x = BatchNormalization()(x) 
        
        x_1 = Conv2D(32, (1, 1),strides=1,padding='same',use_bias=False,kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x)
        x_3 = Conv2D(32, (1, 3),strides=1,padding='same',use_bias=False,kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x)
        x_3 = Conv2D(32, (3, 1),strides=1,padding='same',use_bias=False,kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x_3)
        x_5 = Conv2D(32, (1, 5),strides=1,padding='same',use_bias=False,kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x)
        x_5 = Conv2D(32, (5, 1),strides=1,padding='same',use_bias=False,kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x_5)
        x_7 = Conv2D(32, (1, 7),strides=1,padding='same',use_bias=False,kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x)
        x_7 = Conv2D(32, (7, 1),strides=1,padding='same',use_bias=False,kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x_7)
        x = Concatenate(axis=-1)([x_1,x_3,x_5,x_7])
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Activation("relu")(x)
        x = BatchNormalization()(x) 
        
        x_1 = Conv2D(32, (1, 1),strides=1,padding='same',use_bias=False,kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x)
        x_3 = Conv2D(32, (1, 3),strides=1,padding='same',use_bias=False,kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x)
        x_3 = Conv2D(32, (3, 1),strides=1,padding='same',use_bias=False,kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x_3)
        x_5 = Conv2D(32, (1, 5),strides=1,padding='same',use_bias=False,kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x)
        x_5 = Conv2D(32, (5, 1),strides=1,padding='same',use_bias=False,kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x_5)
        x_7 = Conv2D(32, (1, 7),strides=1,padding='same',use_bias=False,kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x)
        x_7 = Conv2D(32, (7, 1),strides=1,padding='same',use_bias=False,kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x_7)
        x = Concatenate(axis=-1)([x_1,x_3,x_5,x_7])
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Activation("relu")(x)
        x = BatchNormalization()(x) 
        
        x_1 = Conv2D(64, (1, 1),strides=1,padding='same',use_bias=False,kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x)
        x_3 = Conv2D(64, (1, 3),strides=1,padding='same',use_bias=False,kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x)
        x_3 = Conv2D(64, (3, 1),strides=1,padding='same',use_bias=False,kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x_3)
        x_5 = Conv2D(64, (1, 5),strides=1,padding='same',use_bias=False,kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x)
        x_5 = Conv2D(64, (5, 1),strides=1,padding='same',use_bias=False,kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x_5)
        x_7 = Conv2D(64, (1, 7),strides=1,padding='same',use_bias=False,kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x)
        x_7 = Conv2D(64, (7, 1),strides=1,padding='same',use_bias=False,kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x_7)
        x = Concatenate(axis=-1)([x_1,x_3,x_5,x_7])
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Activation("relu")(x)
        x = BatchNormalization()(x) 
        
        
        x_flat = Flatten()(x)
        x_dense = Dense(256,activation="relu",kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x_flat)
        x_dense = Dropout(0.25)(x_dense)
        x_dense = Dense(128,activation="relu",kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x_dense)
        x_dense = Dropout(0.25)(x_dense)
        x_dense = Dense(64,activation="relu",kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x_dense)
        x_dense = Dropout(0.25)(x_dense)
        x_out = Dense(19)(x_dense)
        x_out = Activation('softmax',  name='predictions')(x_out)
        
        
        model = Model(input_image,x_out)
        return model
    
    if model_id==18:
        l1_weight = 1e-6
        l2_weight = 1e-5
        # # # ##canonical
        input_image = Input(shape=(360, 360,1))
        ###### Multiple configureations may the best one win
        
        x = Conv2D(16, (3, 3),strides=1,padding='same',use_bias=False,kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(input_image)
        x = Conv2D(16, (3, 3),strides=1,padding='same',use_bias=False,kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x)
        x = Conv2D(16, (3, 3),strides=1,padding='same',use_bias=False,kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Activation("relu")(x)
        x = BatchNormalization()(x) 
        
        
        x_1 = Conv2D(16, (1, 1),strides=1,padding='same',use_bias=False,kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x)
        x_3 = Conv2D(16, (3, 3),strides=1,padding='same',use_bias=False,kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x)
        x_5 = Conv2D(16, (5, 5),strides=1,padding='same',use_bias=False,kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x)
        x_7 = Conv2D(16, (7, 7),strides=1,padding='same',use_bias=False,kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x)
        x = Concatenate(axis=-1)([x_1,x_3,x_5,x_7])
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Activation("relu")(x)
        x = BatchNormalization()(x) 
        
        x_1 = Conv2D(32, (1, 1),strides=1,padding='same',use_bias=False,kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x)
        x_3 = Conv2D(32, (3, 3),strides=1,padding='same',use_bias=False,kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x)
        x_5 = Conv2D(32, (5, 5),strides=1,padding='same',use_bias=False,kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x)
        x_7 = Conv2D(32, (7, 7),strides=1,padding='same',use_bias=False,kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x)
        x = Concatenate(axis=-1)([x_1,x_3,x_5,x_7])
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Activation("relu")(x)
        x = BatchNormalization()(x) 
        
        x_1 = Conv2D(64, (1, 1),strides=1,padding='same',use_bias=False,kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x)
        x_3 = Conv2D(64, (3, 3),strides=1,padding='same',use_bias=False,kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x)
        x_5 = Conv2D(64, (5, 5),strides=1,padding='same',use_bias=False,kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x)
        x_7 = Conv2D(64, (7, 7),strides=1,padding='same',use_bias=False,kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x)
        x = Concatenate(axis=-1)([x_1,x_3,x_5,x_7])
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Activation("relu")(x)
        x = BatchNormalization()(x) 
        
        x_1 = Conv2D(128, (1, 1),strides=1,padding='same',use_bias=False,kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x)
        x_3 = Conv2D(128, (3, 3),strides=1,padding='same',use_bias=False,kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x)
        x_5 = Conv2D(128, (5, 5),strides=1,padding='same',use_bias=False,kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x)
        x_7 = Conv2D(128, (7, 7),strides=1,padding='same',use_bias=False,kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x)
        x = Concatenate(axis=-1)([x_1,x_3,x_5,x_7])
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Activation("relu")(x)
        x = BatchNormalization()(x) 
        
        
        
        x_flat = Flatten()(x)
        x_dense = Dense(256,activation="relu",kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x_flat)
        x_dense = Dropout(0.25)(x_dense)
        x_dense = Dense(128,activation="relu",kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x_dense)
        x_dense = Dropout(0.25)(x_dense)
        x_dense = Dense(64,activation="relu",kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x_dense)
        x_dense = Dropout(0.25)(x_dense)
        x_out = Dense(19)(x_dense)
        x_out = Activation('softmax',  name='predictions')(x_out)
        
        model = Model(input_image,x_out)
        return model
    if model_id==19:
        x = Conv2D(32, (3, 3),strides=1,padding='same',use_bias=False,kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(input_image)
        x = Conv2D(32, (3, 3),strides=1,padding='same',use_bias=False,kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x)
        x = Conv2D(32, (3, 3),strides=1,padding='same',use_bias=False,kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Activation("relu")(x)
        x = BatchNormalization()(x) 
        
        
        x = Conv2D(64, (3, 3),strides=2,padding='same',use_bias=False,kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x)
        x = Conv2D(64, (3, 3),strides=1,padding='same',use_bias=False,kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x)
        x = Conv2D(64, (3, 3),strides=1,padding='same',use_bias=False,kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Activation("relu")(x)
        x = BatchNormalization()(x) 
        
        
        x = Conv2D(128, (3, 3),strides=2,padding='same',use_bias=False,kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x)
        x = Conv2D(128, (3, 3),strides=1,padding='same',use_bias=False,kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x)
        x = Conv2D(128, (3, 3),strides=1,padding='same',use_bias=False,kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Activation("relu")(x)
        x = BatchNormalization()(x) 
        
        
        
        x = Conv2D(256, (3, 3),strides=1,padding='same',use_bias=False,kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x)
        x = Conv2D(256, (3, 3),strides=1,padding='same',use_bias=False,kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x)
        x = Activation("relu")(x)
        x = BatchNormalization()(x) 


        x_flat = Flatten()(x)
        x_dense = Dense(512,activation="relu",kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x_flat)
        x_dense = Dropout(0.25)(x_dense)
        x_dense = Dense(512,activation="relu",kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x_dense)
        x_dense = Dropout(0.25)(x_dense)
        x_dense = Dense(512,activation="relu",kernel_initializer="truncated_normal",kernel_regularizer=regularizers.l1_l2(l1=l1_weight, l2=l2_weight))(x_dense)
        x_dense = Dropout(0.25)(x_dense)
        x_out = Dense(19)(x_dense)
        x_out = Activation('softmax',  name='predictions')(x_out)
        
        
        model = Model(input_image,x_out)
        return model