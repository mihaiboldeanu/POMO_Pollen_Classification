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
test_path = r"C:\Users\mishu\OneDrive\Desktop\projects\Image classifications\conv\data\test"
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
        image = np.reshape(image[:,:,0],(1,360,360,1))
        file_comp.append(file_name)
        image_stack.append(image)

image_stack = np.concatenate(image_stack,axis=0)        
for i,model in enumerate(models):
    results[i].append( model.predict(image_stack))

final_res = np.sum(results,axis=0)
final_res = final_res[0]

labels_binary = [l if l == "alternaria" else "Other" for l in labels]
labels_unique = sorted(list(set(labels)))
le = LabelEncoder()
le.fit(labels)
labels = le.transform(labels)



cm = confusion_matrix(labels,np.argmax(final_res,axis=1))
clas_rep = classification_report(labels,np.argmax(final_res,axis=1))
plot_conf_matrix(cm,labels_unique,"conf matrix2")




labels_unique = sorted(list(set(labels_binary)))
le = LabelEncoder()
le.fit(labels_binary)
labels = le.transform(labels_binary)

final_res_binary = np.argmax(final_res,axis=1)
final_res_binary = [l if l==1 else 0 for l  in final_res_binary]
cm = confusion_matrix(labels,final_res_binary)
clas_rep = classification_report(labels,final_res_binary)
plot_conf_matrix(cm,labels_unique,"conf matrix_binary")
