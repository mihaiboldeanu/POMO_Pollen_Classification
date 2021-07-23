# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 07:40:17 2021

@author: Mihai Boldeanu
"""
import gc
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras import backend as K
from skimage.morphology import dilation, disk
import lib_pomo


# Free up RAM in case the model definition cells were run multiple times
#tf.keras.backend.clear_session()

ret_list = [["filename",
             "coordinates",
             "cnn confirmed(1-alternaria;0-could not check;11-might be junk)"]]

path = r"C:\Users\Mihai Boldeanu\Desktop\pollen_clas\new data\different cases"


img_size = (960, 1280)
num_classes = 1
batch_size = 2

img_paths = lib_pomo.get_unique_pictures(path)

model_u_net = load_model("u_net_classical-0.00935.hdf5")
model_cnn = load_model("cnn_model-0-aug-0.93307.hdf5")
x_unet = np.zeros((1,960,1280,1))
for i,key  in enumerate(list(img_paths.keys())):

    print (i)
    file_name = img_paths[key][0]["file"]
    file_path = os.path.join(path,file_name)
    img = load_img(file_path,
                   target_size=img_size,
                   color_mode="grayscale")
    temp_x = lib_pomo.augment(img,0,0,0)
    temp_x = temp_x/255.0
    x_unet[0,:,:,0] = temp_x
    x_unet_tensor = tf.convert_to_tensor(x_unet, dtype=tf.float32)
    result_acc = model_u_net.predict(x_unet_tensor)

    expanded_mask = lib_pomo.expand_mask(result_acc)

    if (expanded_mask==2).any():

        alternaria = lib_pomo.get_one_class(expanded_mask)
        coord_list = lib_pomo.segment_image(alternaria)
        coord_list_valid = []
        for coordinate in coord_list:
            new_img =  np.zeros((1,960,1280,1))
            corners = coordinate
            
            area = alternaria[:,corners[0]:corners[1],corners[2]:corners[3],:]

            if lib_pomo.small(coordinate,area):
                expanded_mask[:,corners[0]:corners[1],corners[2]:corners[3],:][np.where(expanded_mask[:,corners[0]:corners[1],corners[2]:corners[3],:]==2)] = 1
                continue

            if lib_pomo.big(coordinate,area):
                name = os.path.split(file_path)[1]
                ret_list.append([name,corners,0])  
                continue

            coord_list_valid.append(coordinate)


        if coord_list_valid:
            x=np.zeros((len(coord_list_valid),360,360,1))
            for j,corners in enumerate(coord_list_valid):
                crop_for_cnn = lib_pomo.get_crop(img,corners)
                mask_for_cnn = lib_pomo.get_crop(alternaria[0,:,:,0],corners)
                mask_for_cnn[:,:,0] = dilation(mask_for_cnn[:,:,0],selem=disk(4))
                mask_for_cnn = mask_for_cnn > 0
                x[j,:,:,:] = crop_for_cnn
            x = x/255.
            x_cnn_tensor = tf.convert_to_tensor(x, dtype=tf.float32) 
            confirmation = model_cnn.predict(x_cnn_tensor)
            confirmation = np.argmax(confirmation,axis=1)

            name = os.path.split(file_path)[1]
            for c,conf in enumerate(confirmation):
                ret_list.append([name,coord_list_valid[c],conf])

    lib_pomo.save_image(x_unet,expanded_mask,file_path)
    K.clear_session()
    _ = gc.collect()

name =  os.path.join(path,"metadata.log")
with open(name,"w") as f:
    for line in ret_list:
        text_line = [str(l) for l in line]
        text_line = ",".join(text_line)+"\n"
        f.write(text_line)
