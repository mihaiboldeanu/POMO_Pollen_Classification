# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 11:40:48 2020

@author: mishu
"""
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 12:39:22 2019

@author: mishu
"""
import os
import random
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
#from tensorflow import keras
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping,ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import OrderedEnqueuer


import lib_pomo

in_dir = r"C:\Users\Mihai Boldeanu\Desktop\pollen_clas\all_alternaria_maskRCNN\input"
t_dir1 = r"C:\Users\Mihai Boldeanu\Desktop\pollen_clas\all_alternaria_maskRCNN\output1"
t_dir2 = r"C:\Users\Mihai Boldeanu\Desktop\pollen_clas\all_alternaria_maskRCNN\output2"
img_size = (960, 1280)
num_classes = 1
batch_size = 6

in_path = [os.path.join(in_dir, f) for f in os.listdir(in_dir) if f.endswith(".png")]
t_path1 = [os.path.join(t_dir1, f) for f in os.listdir(t_dir1) if f.endswith(".png")]
t_path2 = [os.path.join(t_dir2, f) for f in os.listdir(t_dir2) if f.endswith(".png")]

input_img_paths = sorted(in_path)
target_img_paths1 = sorted(t_path1)
target_img_paths2 = sorted(t_path2)

# Split our img paths into a training and a validation set
val_samples = 300

random.Random(1337).shuffle(input_img_paths)
random.Random(1337).shuffle(target_img_paths1)
random.Random(1337).shuffle(target_img_paths2)

train_input_img_paths = input_img_paths[:-val_samples]
train_target_img_paths1 = target_img_paths1[:-val_samples]
train_target_img_paths2 = target_img_paths2[:-val_samples]

val_input_img_paths = input_img_paths[-val_samples:]
val_target_img_paths1 = target_img_paths1[-val_samples:]
val_target_img_paths2 = target_img_paths2[-val_samples:]


# Instantiate data Sequences for each split
train_gen = lib_pomo.Pollen(batch_size, img_size,
                            train_input_img_paths,
                            train_target_img_paths1,
                            train_target_img_paths2,augment=True)
val_gen = lib_pomo.Pollen(batch_size, img_size,
                          val_input_img_paths,
                          val_target_img_paths1,
                          val_target_img_paths2,
                          augment=False)
# Free up RAM in case the model definition cells were run multiple times
tf.keras.backend.clear_session()

# Build model
model = lib_pomo.get_model((960,1280,1), 3)
#model.summary()

adam = Adam(learning_rate=0.00001,
            beta_1=0.6,
            beta_2=0.999,
            amsgrad=False,
            clipnorm=1) 
model.compile(loss='sparse_categorical_crossentropy',
              optimizer=adam,
              metrics=["sparse_categorical_accuracy"])

# checkpoint
filepath="u_net_classical-{val_loss:.5f}.hdf5"
checkpoint = ModelCheckpoint(filepath,
                             monitor='val_loss',
                             verbose=1,
                             save_best_only=True,
                             mode='min')
reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                              mode='min',
                              verbose = 1,
                              factor=0.5,
                              patience=10,
                              cooldown=2,
                              min_lr=0.0000001)
es = EarlyStopping(monitor='val_loss',
                   mode='min',
                   verbose=1,
                   patience=100)
callbacks_list = [checkpoint,reduce_lr,es]

model_old = load_model("u_net_classical-0.01089.hdf5") 
model.set_weights(model_old.get_weights())


enq_real = OrderedEnqueuer(train_gen, use_multiprocessing=False)
enq_real.start(workers=6, max_queue_size=60)

enq_val_real = OrderedEnqueuer(val_gen, use_multiprocessing=False)
enq_val_real.start(workers=6, max_queue_size=20)

gen_real = enq_real.get()
val_real = enq_val_real.get()
history = model.fit(gen_real,
                    epochs=500,
                    steps_per_epoch=len(train_input_img_paths)//batch_size,
                    callbacks=callbacks_list,
                    verbose=1,
                    shuffle=True,
                    validation_data=val_real,
                    validation_steps=len(val_input_img_paths)//batch_size,
                    use_multiprocessing=False,  # CHANGED
                    max_queue_size=10,
                    workers=1)
enq_real.stop()
val_real.stop()

# model_accurate = load_model("new_model-0.00819.hdf5")   
# model_precise = load_model("new_model-0.00938.hdf5")   
# x = np.zeros((1,960,1280,1))
# for i,path  in enumerate(val_input_img_paths):

#     img = load_img(path, 
#                    target_size=img_size,
#                    color_mode="grayscale")
#     temp_x = lib_pomo.augment(img,0,0,0)
#     temp_x = temp_x/255.0
#     x[0,:,:,0] = temp_x
#     tensor = tf.convert_to_tensor(x, dtype=tf.float32)
#     result_acc = model_accurate.predict_on_batch(x)
#     result_prec =  model_precise.predict_on_batch(x)

#     result = (result_acc + result_prec)/2.
#     result = np.maximum(result_acc,result_prec)
#     #result = result_acc
#     expanded_mask = lib_pomo.expand_mask(result)
#     name = str(i)+"new+old"
#     lib_pomo.plot_original_mask(x,expanded_mask,name)
#     break

    
    