# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 07:45:38 2021

@author: Mihai Boldeanu
"""

from tensorflow.keras import regularizers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Activation, Conv2DTranspose
from tensorflow.keras.layers import Input, BatchNormalization,Dropout
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.layers import add,concatenate



def get_model(img_size, num_classes,first_layer = 16):
    inputs = Input(shape=img_size )

    ### [First half of the network: downsampling inputs] ###
    l1_weight = 1e-6 * 16./first_layer
    l2_weight = 1e-5 * (16./first_layer)**2
    # Entry block
    x = Conv2D(first_layer, 3, strides=2, padding="same",use_bias=False,
               kernel_initializer='glorot_normal')(inputs)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    previous_block_activation = x  # Set aside residual

    # Blocks 1, 2, 3 are identical apart from the feature depth.
    for filters in [ 2*first_layer, 4*first_layer,8*first_layer,16*first_layer]:
        x = Activation("relu")(x)
        x = Conv2D(filters, 3, padding="same",use_bias=False,
                   kernel_initializer='glorot_normal',)(x)
        x = BatchNormalization()(x)

        x = Activation("relu")(x)
        x = Conv2D(filters, 3, padding="same",use_bias=False,
                   kernel_initializer='glorot_normal')(x)
        x = BatchNormalization()(x)

        x = MaxPooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual = Conv2D(filters, 1, strides=2, padding="same",use_bias=False,
                          kernel_initializer='glorot_normal')(
            previous_block_activation)
        x = add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    ### [Second half of the network: upsampling inputs] ###

    for filters in [ 16*first_layer, 8*first_layer,4*first_layer,2*first_layer,first_layer]:
        x = Activation("relu")(x)
        x = Conv2DTranspose(filters, 3, padding="same",use_bias=False,
                            kernel_initializer='glorot_normal')(x)
        x = BatchNormalization()(x)

        x = Activation("relu")(x)
        x = Conv2DTranspose(filters, 3, padding="same",use_bias=False,
                            kernel_initializer='glorot_normal')(x)
        x = BatchNormalization()(x)

        x = UpSampling2D(2)(x)

        # Project residual
        residual = UpSampling2D(2)(previous_block_activation)
        residual = Conv2D(filters, 1, padding="same",use_bias=False,
                          kernel_initializer='glorot_normal')(residual)
        x = add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    # Add a per-pixel classification layer
    outputs = Conv2D(num_classes, 3, activation="softmax", padding="same",
                      kernel_initializer='glorot_normal')(x)

    # Define the model
    model = Model(inputs, outputs)
    return model




def get_model_v2(img_size, num_classes,first_layer = 16):
    inputs = Input(shape=img_size )

    ### [First half of the network: downsampling inputs] ###
    l1_weight = 1e-6 * 16./first_layer
    l2_weight = 1e-5 * (16./first_layer)**2
    previous_block_activations = []
    # Entry block
    x = Conv2D(first_layer, 3, strides=2, padding="same",use_bias=False,
               kernel_initializer='glorot_normal')(inputs)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    previous_block_activation = x  # Set aside residual
    previous_block_activations.append(x)  # Set aside deep residual
    # Blocks 1, 2, 3 are identical apart from the feature depth.
    for filters in [ 2*first_layer, 4*first_layer,8*first_layer,16*first_layer]:
        x = Activation("relu")(x)
        x = Conv2D(filters, 3, padding="same",use_bias=False,
                   kernel_initializer='glorot_normal')(x)
        x = BatchNormalization()(x)

        x = Activation("relu")(x)
        x = Conv2D(filters, 3, padding="same",use_bias=False,
                   kernel_initializer='glorot_normal')(x)
        x = BatchNormalization()(x)

        x = MaxPooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual = Conv2D(filters, 1, strides=2, padding="same",use_bias=False,
                          kernel_initializer='glorot_normal')(
            previous_block_activation)
        previous_block_activations.append(x)
        x = add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual
        


    ### [Second half of the network: upsampling inputs] ###
    previous_block_activations.reverse()
    for i,filters in enumerate([ 16*first_layer, 8*first_layer,4*first_layer,2*first_layer,first_layer]):
        x = Activation("relu")(x)
        x = Conv2DTranspose(filters, 3, padding="same",use_bias=False,
                            kernel_initializer='glorot_normal')(x)
        x = BatchNormalization()(x)

        x = Activation("relu")(x)
        x = Conv2DTranspose(filters, 3, padding="same",use_bias=False,
                            kernel_initializer='glorot_normal')(x)
        x = BatchNormalization()(x)

        x = UpSampling2D(2)(x)

        # Project residual
        residual = UpSampling2D(2)(previous_block_activation)
        residual = Conv2D(filters, 1, padding="same",use_bias=False,
                          kernel_initializer='glorot_normal')(residual)
        x = add([x, residual])  # Add back residual
        deep_residual = UpSampling2D(2)(previous_block_activations[i])
        deep_residual = Conv2D(filters, 1, padding="same",
                               kernel_initializer='glorot_normal')(deep_residual)
        x = concatenate([x, deep_residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    # Add a per-pixel classification layer
    outputs = Conv2D(num_classes, 3, activation="softmax", padding="same",
                     kernel_initializer='glorot_normal')(x)

    # Define the model
    model = Model(inputs, outputs)
    return model
def get_model_unet(img_size, num_classes,first_layer = 16):
    l1_weight = 1e-6 * 16./first_layer
    l2_weight = 1e-5 * (16./first_layer)**2
    inputs = Input(shape=img_size )

    ### [First half of the network: downsampling inputs] ###
    previous_block_activations = []
    # Entry block
    x = Conv2D(first_layer, 3, strides=2, padding="same",use_bias=False,
               kernel_initializer='glorot_normal')(inputs)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    previous_block_activations.append(x)  # Set aside residual

    # Blocks 1, 2, 3 are identical apart from the feature depth.
    for filters in [ 2*first_layer, 4*first_layer,8*first_layer,16*first_layer]:
        x = Activation("relu")(x)
        x = Conv2D(filters, 3, padding="same",use_bias=False,
                   kernel_initializer='glorot_normal')(x)
        x = BatchNormalization()(x)

        x = Activation("relu")(x)
        x = Conv2D(filters, 3, padding="same",use_bias=False,
                   kernel_initializer='glorot_normal')(x)
        x = BatchNormalization()(x)

        x = MaxPooling2D(3, strides=2, padding="same")(x)

        # # Project residual
        # residual = Conv2D(filters, 1, strides=2, padding="same",kernel_initializer='glorot_normal')(
        #     previous_block_activation)
        # x = concatenate([x, residual])  # Add back residual
        previous_block_activations.append(x) # Set aside next residual

    ### [Second half of the network: upsampling inputs] ###
    previous_block_activations.reverse()
    for i,filters in enumerate([ 16*first_layer, 8*first_layer,4*first_layer,2*first_layer,first_layer]):
        x = Activation("relu")(x)
        x = Conv2DTranspose(filters, 3, padding="same",use_bias=False,
                            kernel_initializer='glorot_normal')(x)
        x = BatchNormalization()(x)

        x = Activation("relu")(x)
        x = Conv2DTranspose(filters, 3, padding="same",use_bias=False,
                            kernel_initializer='glorot_normal')(x)
        x = BatchNormalization()(x)

        x = UpSampling2D(2)(x)

        # Project residual
        
        residual = UpSampling2D(2)(previous_block_activations[i])
        residual = Conv2D(filters, 1, padding="same",kernel_initializer='glorot_normal')(residual)
        x = concatenate([x, residual])  # Add back residual
        # previous_block_activation = x  # Set aside next residual

    # Add a per-pixel classification layer
    outputs = Conv2D(num_classes, 3, activation="softmax", padding="same",
                     kernel_initializer='glorot_normal')(x)

    # Define the model
    model = Model(inputs, outputs)
    return model

def get_model_unet_add(img_size, num_classes,first_layer = 16):
    l1_weight = 1e-6 * 16./first_layer
    l2_weight = 1e-5 * (16./first_layer)**2
    inputs = Input(shape=img_size )

    ### [First half of the network: downsampling inputs] ###
    previous_block_activations = []
    # Entry block
    x = Conv2D(first_layer, 3, strides=2, padding="same",use_bias=False,
               kernel_initializer='glorot_normal')(inputs)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    previous_block_activations.append(x)  # Set aside residual

    # Blocks 1, 2, 3 are identical apart from the feature depth.
    for filters in [ 2*first_layer, 4*first_layer,8*first_layer,16*first_layer]:
        x = Activation("relu")(x)
        x = Conv2D(filters, 3, padding="same",use_bias=False,
                   kernel_initializer='glorot_normal')(x)
        x = BatchNormalization()(x)

        x = Activation("relu")(x)
        x = Conv2D(filters, 3, padding="same",use_bias=False,
                   kernel_initializer='glorot_normal')(x)
        x = BatchNormalization()(x)

        x = MaxPooling2D(3, strides=2, padding="same")(x)

        # # Project residual
        # residual = Conv2D(filters, 1, strides=2, padding="same",kernel_initializer='glorot_normal')(
        #     previous_block_activation)
        # x = concatenate([x, residual])  # Add back residual
        previous_block_activations.append(x) # Set aside next residual

    ### [Second half of the network: upsampling inputs] ###
    previous_block_activations.reverse()
    for i,filters in enumerate([ 16*first_layer, 8*first_layer,4*first_layer,2*first_layer,first_layer]):
        x = Activation("relu")(x)
        x = Conv2DTranspose(filters, 3, padding="same",use_bias=False,
                            kernel_initializer='glorot_normal')(x)
        x = BatchNormalization()(x)

        x = Activation("relu")(x)
        x = Conv2DTranspose(filters, 3, padding="same",use_bias=False,
                            kernel_initializer='glorot_normal')(x)
        x = BatchNormalization()(x)

        x = UpSampling2D(2)(x)

        # Project residual
        
        residual = UpSampling2D(2)(previous_block_activations[i])
        residual = Conv2D(filters, 1, padding="same",kernel_initializer='glorot_normal')(residual)
        x = add([x, residual])  # Add back residual
        # previous_block_activation = x  # Set aside next residual

    # Add a per-pixel classification layer
    outputs = Conv2D(num_classes, 3, activation="softmax", padding="same",
                     kernel_initializer='glorot_normal')(x)

    # Define the model
    model = Model(inputs, outputs)
    return model

def get_model_add(img_size, num_classes,first_layer = 16):
    inputs = Input(shape=img_size )

    ### [First half of the network: downsampling inputs] ###
    previous_block_activations = []
    # Entry block
    x = Conv2D(first_layer, 3, strides=2, padding="same",use_bias=False,kernel_initializer='glorot_normal')(inputs)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    # previous_block_activations.append(x)  # Set aside residual
    previous_block_activation = x
    # Blocks 1, 2, 3 are identical apart from the feature depth.
    for filters in [ 2*first_layer, 4*first_layer,8*first_layer,16*first_layer]:
        x = Activation("relu")(x)
        x = Conv2D(filters, 3, padding="same",use_bias=False,kernel_initializer='glorot_normal')(x)
        x = BatchNormalization()(x)

        x = Activation("relu")(x)
        x = Conv2D(filters, 3, padding="same",use_bias=False,kernel_initializer='glorot_normal')(x)
        x = BatchNormalization()(x)

        x = MaxPooling2D(3, strides=2, padding="same")(x)

        # # Project residual
        residual = Conv2D(filters, 1, strides=2, padding="same",kernel_initializer='glorot_normal')(
                previous_block_activation)
        x = add([x, residual])  # Add back residual
        # previous_block_activations.append(x) # Set aside next residual
        previous_block_activation = x

    ### [Second half of the network: upsampling inputs] ###
    previous_block_activations.reverse()
    for i,filters in enumerate([ 16*first_layer, 8*first_layer,4*first_layer,2*first_layer,first_layer]):
        x = Activation("relu")(x)
        x = Conv2DTranspose(filters, 3, padding="same",use_bias=False,kernel_initializer='glorot_normal')(x)
        x = BatchNormalization()(x)

        x = Activation("relu")(x)
        x = Conv2DTranspose(filters, 3, padding="same",use_bias=False,kernel_initializer='glorot_normal')(x)
        x = BatchNormalization()(x)

        x = UpSampling2D(2)(x)

        # Project residual
        
        # residual = UpSampling2D(2)(previous_block_activations[i])
        # residual = Conv2D(filters, 1, padding="same",kernel_initializer='glorot_normal')(residual)
        # x = concatenate([x, residual])  # Add back residual
        # # previous_block_activation = x  # Set aside next residual
        
        residual = UpSampling2D(2)(previous_block_activation)
        residual = Conv2D(filters, 1, padding="same",kernel_initializer='glorot_normal')(residual)
        x = concatenate([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    # Add a per-pixel classification layer
    output_1 = Conv2D(num_classes, 3, activation="softmax", padding="same",name="class_segmentation")(x)
    output = Conv2D(20, 3, activation="relu", padding="same")(output_1)
    output = Conv2D(20, 3, activation="relu", padding="same")(output)
    output_2 = Conv2D(1, 3, activation="sigmoid", padding="same",name="instance_segmentation")(output)
    # Define the model
    model = Model(inputs, [output_1,output_2])
    return model



def get_model_full(img_size, num_classes,first_layer = 16,
                  deep_residual_block=False,deep_residual_operation="add",
                  residual_block=False,residual_operation="add"):
    inputs = Input(shape=img_size )

    ### [First half of the network: downsampling inputs] ###
    l1 = 0*1e-7 
    l2 = 0*1e-6 
    previous_block_activations = []
    # Entry block
    x = Conv2D(first_layer, 3,  padding="same",use_bias=False,
               kernel_initializer='glorot_normal',
               kernel_regularizer=regularizers.l1_l2(l1=l1, l2=l2))(inputs)
    x = Activation("relu")(x)
    x = BatchNormalization()(x)
    # x = Dropout(0.05)(x)
    
    x = Conv2D(first_layer, 3, strides=2, padding="same",use_bias=False,
               kernel_initializer='glorot_normal',
               kernel_regularizer=regularizers.l1_l2(l1=l1, l2=l2))(x)
    x = Activation("relu")(x)
    x = BatchNormalization()(x)
    # x = Dropout(0.05)(x)
    
    

    if residual_block :
        previous_block_activation = x  # Set aside residual
    if  deep_residual_block:
        previous_block_activations.append(x)  # Set aside deep residual
    # Blocks 1, 2, 3 are identical apart from the feature depth.
    for filters in [ 2*first_layer, 4*first_layer,8*first_layer]:
        
        x = Conv2D(filters, 3, padding="same",use_bias=False,
                   kernel_initializer='glorot_normal',
               kernel_regularizer=regularizers.l1_l2(l1=l1, l2=l2))(x)
        
        x = Activation("relu")(x)
        x = BatchNormalization()(x)
        # x = Dropout(0.05)(x)
        

        
        x = Conv2D(filters, 3, padding="same",use_bias=False,
                   kernel_initializer='glorot_normal',
               kernel_regularizer=regularizers.l1_l2(l1=l1, l2=l2))(x)
        x = MaxPooling2D(3, strides=2, padding="same")(x)
        x = Activation("relu")(x)
        x = BatchNormalization()(x)
        # x = Dropout(0.05)(x)
        

        
        
        if deep_residual_block:
            previous_block_activations.append(x)
        # Project residual
        if residual_block :
            residual = Conv2D(filters, 1, strides=2, padding="same",use_bias=False,
                              kernel_initializer='glorot_normal',
               kernel_regularizer=regularizers.l1_l2(l1=l1, l2=l2))(
                previous_block_activation)
            
            if residual_operation =="add":
                x = add([x, residual])  # Add back residual
            else:
                x = concatenate([x, residual])  # Add back residual
            previous_block_activation = x  # Set aside next residual
        


    ### [Second half of the network: upsampling inputs] ###
    if  deep_residual_block:
        previous_block_activations.reverse()
        previous_block_activations = previous_block_activations[1:]
    for i,filters in enumerate([  8*first_layer,4*first_layer,2*first_layer,first_layer]):
        
        x = Conv2DTranspose(filters, 3, padding="same",use_bias=False,
                            kernel_initializer='glorot_normal',
               kernel_regularizer=regularizers.l1_l2(l1=l1, l2=l2))(x)
        x = Activation("relu")(x)
        x = BatchNormalization()(x)
        # x = Dropout(0.05)(x)
        

        
        x = Conv2DTranspose(filters, 3, padding="same",use_bias=False,
                            kernel_initializer='glorot_normal',
               kernel_regularizer=regularizers.l1_l2(l1=l1, l2=l2))(x)
        x = Activation("relu")(x)
        x = BatchNormalization()(x)
        # x = Dropout(0.05)(x)
        
        x = UpSampling2D(2)(x)

        # Project residual
        if residual_block :
            residual = UpSampling2D(2)(previous_block_activation)
            residual = Conv2D(filters, 1, padding="same",use_bias=False,
                              kernel_initializer='glorot_normal',
               kernel_regularizer=regularizers.l1_l2(l1=l1, l2=l2))(residual)
            if residual_operation =="add":
                x = add([x, residual])  # Add back residual
            else:
                x = concatenate([x, residual])  # Add back residual
            
        if deep_residual_block and i < len(previous_block_activations) :
            # deep_residual = UpSampling2D(2)(previous_block_activations[i])
            deep_residual = Conv2D(filters, 1, padding="same",use_bias=False,
                                   kernel_initializer='glorot_normal',
               kernel_regularizer=regularizers.l1_l2(l1=l1, l2=l2))(previous_block_activations[i])
            if deep_residual_operation =="add":
                x = add([x, deep_residual])  # Add back residual
            else:
                x = concatenate([x, deep_residual])  # Add back residual
            
        previous_block_activation = x  # Set aside next residual

    # Add a per-pixel classification layer
    outputs = Conv2D(num_classes, 3, activation="softmax", padding="same",
                     kernel_initializer='glorot_normal',
               kernel_regularizer=regularizers.l1_l2(l1=l1, l2=l2))(x)

    # Define the model
    model = Model(inputs, outputs)
    return model