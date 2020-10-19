# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 18:47:14 2019

@author: mishu
"""

import os
import PIL.Image
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,Dropout,Dense,MaxPooling2D,Flatten
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
import numpy as np
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.preprocessing import LabelEncoder
from shutil import copyfile

def plot_conf_matrix(conf_matrix,labels,title,percentage=True):
    fig, ax = plt.subplots(figsize=(10,10))
    norm = np.sum(conf_matrix,axis=1)
    one = np.ones(conf_matrix.shape)
    one_norm = one *norm
    one_norm = one_norm.T
    if percentage:
        to_plot = conf_matrix/one_norm
        to_plot = np.around(to_plot,2)
    else:
        to_plot = conf_matrix
        
    ax.imshow(to_plot)
    # We want to show all ticks...
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    # ... and label them with the respective list entries
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    
    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
    
    # Loop over data dimensions and create text annotations.
    for i in range(len(labels)):
        for j in range(len(labels)):
            if to_plot[i,j]>0.5:
                c= "k"
            else:
                c="w"
            ax.text(j, i, to_plot[i, j],
                    ha="center", va="center", color=c)
    
    ax.set_title("Confusion matrix")
    fig.tight_layout()
    

    plt.savefig(title+".png")

path = r"C:\Users\mishu\OneDrive\Desktop\projects\Image classifications\conv\conv_net\models"
model_list = os.listdir(path)
models = []
for file_name in model_list:
    file_path = os.path.join(path,file_name)
    models.append(load_model(file_path))

file_comp = []
test_path = r"C:\Users\mishu\OneDrive\Desktop\projects\Image classifications\conv\conv_net\Pomo_crop"
list_of_folders = os.listdir(test_path)
results = [[] for i in range(len(models))]



labels = []
image_stack = []
for folder in list_of_folders:
    folder_path = os.path.join(test_path,folder)
    list_of_files = os.listdir(folder_path)
    for file_name in list_of_files:
        if "ini" in file_name:
            continue
        labels.append(folder)
        file_path = os.path.join(folder_path,file_name)

        image = plt.imread(file_path)
        im_shape = image.shape
        height = 360
        width = 360
        new_image = np.zeros((360,360,1))
        new_image[int(180-im_shape[0]/2):int(180+im_shape[0]/2),int(180-im_shape[1]/2):int(180+im_shape[1]/2),:] = image[:,:,0:1]
        
        file_comp.append(file_path)
        image_stack.append(new_image)

image_stack = np.stack(image_stack)        
for i,model in enumerate(models):
    results[i].append( model.predict(image_stack))

final_res = np.mean(results,axis=0)
final_res = final_res[0]
final_res_max = np.argmax(final_res,axis=1)
label_list = ['Alnus', 'Alternaria', 'Betula','Carpinus', 'Corylus', 'Fungus', 'Junk', 'Pinus','Poacea', 'Varia']
label = [label_list[i] for i in final_res_max]

save_path = r"C:\Users\mishu\OneDrive\Desktop\projects\Image classifications\conv\conv_net\Pomo_crop"
for index,val in enumerate(final_res_max):
    l = label_list[val]
    save_path = file_comp[index].replace("crop_",l+"_crop_").replace("Pomo_crop","Pomo_crop_classified")
    copyfile(file_comp[index],save_path)