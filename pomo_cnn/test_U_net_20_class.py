# -*- coding: utf-8 -*-
"""
Created on Tue Aug 31 12:09:41 2021

@author: Mihai Boldeanu
"""

import os
import numpy as np
from scipy.ndimage import rotate
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import pomonet

path_validation = r"F:\pollen_clas\data\validation"
folders = os.listdir(path_validation)
labels = []
validation_images = []
for label,folder in enumerate(folders):
    folder_path = os.path.join(path_validation,folder)
    files = os.listdir(folder_path)
    for file in files:
        labels.append(label)
        validation_images.append(os.path.join(folder_path,file))
        
        
def read_and_reshape(image_path):
    background = np.zeros((960,1280))
    image = np.array(load_img(image_path,
                              target_size=(360,360),
                              color_mode="grayscale"))
    image_flip_lr = np.flip(image,axis=1)
    image_flip_ud = np.flip(image,axis=0)
    image_flip = image.T
    image_rotate_45 = np.around(rotate(image,45,reshape=False))
    image_rotate_270 = np.around(rotate(image,270,reshape=False))
    
    background[20:380,:360] += image
    background[20:380,400:760] += image_flip_lr
    background[20:380,780:1140] += image_flip_ud
    
    background[400:760,:360] += image_flip
    background[400:760,380:740] += image_rotate_45
    background[400:760,740:1100] += image_rotate_270
    
    return background/255.

dependencies = {'macro_IOU': pomonet.utils.macro_IOU,
                'bce_dice_iou_loss': pomonet.utils.bce_dice_iou_loss,
                'iou_coef': pomonet.utils.iou_coef,
                'dice_coef': pomonet.utils.dice_coef,}

model = load_model("best_u_net_home-d_res-add-32.hdf5",custom_objects=dependencies)
bins = np.array(range(21))
labels_hat = []
for i,image_path in enumerate(validation_images):

    if i %100 ==0:
        print(i)
    image_og = np.array(load_img(image_path,
                                  target_size=(360,360),
                                  color_mode="grayscale"))
    image_big = read_and_reshape(image_path)
    
    # plt.figure(dpi=600)
    # plt.imshow(image_og,cmap="gray")
    
    # plt.figure(dpi=600)
    # plt.imshow(image_big,cmap="gray")
    
    
    x = np.reshape(image_big,(1,960,1280,1))
    y_og = model(x)
    y = np.sum(y_og[0],axis=(0,1))
    prediction = np.argmax(y[1:])
    # plt.figure(dpi=600)
    # plt.imshow(np.argmax(y_og,axis=-1)[0])
    # y_hist = np.histogram(y.ravel(),bins=bins)
    # y_hist = y_hist[0]
    # y_hist = y_hist[1:]
    # prediction = np.argmax(y_hist) 
    labels_hat.append(prediction)
    
    # plt.figure()
    # plt.imshow(np.argmax(y_og[0],axis=-1),vmin=0,vmax=20.01)
    
print('Confusion Matrix')
labels_name = [f.split("\\")[4] for f in validation_images]
labels_name = sorted(list(set(labels_name)))
conf = confusion_matrix(labels, labels_hat,normalize='true')
plt.imshow(conf,vmin=0,vmax=1.)
print('Classification Report')
target_names = list(range(19))
print(classification_report(labels, labels_hat, target_names=labels_name))
