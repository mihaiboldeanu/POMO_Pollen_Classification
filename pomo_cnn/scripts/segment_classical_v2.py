# -*- coding: utf-8 -*-
"""
Created on Fri Jul 24 16:43:37 2020

@author: mishu
"""

import os
import PIL.Image
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
from tensorflow.keras.models import load_model
import matplotlib.patches as patches
import skimage.color
import skimage.filters
import skimage.io
import skimage.viewer
import skimage.measure
import skimage.color
from skimage.morphology import erosion, dilation, opening, closing, white_tophat,black_tophat
from skimage.morphology import disk


def segment_image_2(image):


    blur = skimage.filters.gaussian(image, sigma=0.7)### Slight blur to help image segmentation
    mask = blur > 0.1
    mask = erosion(mask,selem=disk(2))
    labeled_image = skimage.measure.label(mask, connectivity=1., return_num=True)### Actual segmentation CCA
    
    values_mask = labeled_image[0]### Get a list of segments
    plt.figure(figsize=(10,12))
    plt.imshow(values_mask,cmap="viridis")
    plt.show()
    plt.figure(figsize=(10,12))
    plt.imshow(image,cmap="gray")
    plt.show()
    coord_list = []
    image_list = []
    crop_list = []
    image_modified = np.copy(image)
    uniques = np.unique(values_mask)
    for uniq in uniques:
        image_temp = np.zeros(image.shape)
        indexex = np.where(values_mask==uniq)
        image_temp[indexex] = image[indexex]

        x_min = np.min(indexex[0])-2
        x_max = np.max(indexex[0])+2
        y_min = np.min(indexex[1])-2
        y_max = np.max(indexex[1])+2
        crop_im = image_temp[x_min:x_max,y_min:y_max]
        im_shape = crop_im.shape
        
        if im_shape[0]>250 or im_shape[1]>250:
            continue
        if im_shape[0]<8 or im_shape[1]<8 or len(np.where(values_mask==uniq)[0]) < 64:
            image_modified[x_min:x_max,y_min:y_max] = 0
            continue
        
        new_image = np.zeros((360,360,1))
        new_image[int(180-im_shape[0]/2):int(180+im_shape[0]/2),int(180-im_shape[1]/2):int(180+im_shape[1]/2),0] = crop_im
        image_modified[x_min:x_max,y_min:y_max] = 0
        image_list.append(new_image)
        crop_list.append(crop_im)
        coord_list.append((x_min,x_max,y_min,y_max))
    
    plt.figure(figsize=(10,12))
    plt.imshow(image_modified,cmap="gray")
    plt.show()
    if np.sum(image_modified)==0:
        return crop_list,image_list,coord_list
    else:
        image_modified_2 = np.copy(image_modified)
        blur = skimage.filters.gaussian(image_modified_2, sigma=1.5)### Slight blur to help image segmentation
        
        blur_mask = np.copy(image_modified_2)
        blur_mask[np.where(blur_mask==0 )] = 140/255. 
        mask = blur_mask < np.mean(blur[np.where(blur>0)])
        mask = erosion(mask,selem=disk(2))

        labeled_image = skimage.measure.label(mask, connectivity=1., return_num=True)### Actual segmentation CCA
        values_mask = labeled_image[0]
        
        plt.figure(figsize=(10,12))
        plt.imshow(values_mask,cmap="viridis")
        plt.show()
        plt.figure(figsize=(10,12))
        plt.imshow(image_modified_2,cmap="gray")
        plt.show()
        uniques = np.unique(values_mask)
        for uniq in uniques:
            image_temp = np.zeros(image.shape)
            indexex = np.where(values_mask==uniq)
            image_temp[indexex] = image_modified[indexex]
            
            x_min = np.min(indexex[0])-2
            x_max = np.max(indexex[0])+2
            y_min = np.min(indexex[1])-2
            y_max = np.max(indexex[1])+2
            crop_im = image[x_min:x_max,y_min:y_max]
            im_shape = crop_im.shape
            
            if im_shape[0]>360 or im_shape[1]>360:
                continue
            if im_shape[0]<20 or im_shape[1]<20 or len(np.where(values_mask==uniq)[0]) < 64:
                image_modified[x_min:x_max,y_min:y_max] = 0
                continue
            # plt.figure(figsize=(10,12))
            # plt.imshow(crop_im,cmap="gray")
            new_image = np.zeros((360,360,1))
            new_image[int(180-im_shape[0]/2):int(180+im_shape[0]/2),int(180-im_shape[1]/2):int(180+im_shape[1]/2),0] = crop_im
            image_modified_2[x_min:x_max,y_min:y_max] = 0
            image_list.append(new_image)
            crop_list.append(crop_im)
            coord_list.append((x_min,x_max,y_min,y_max))

        return crop_list,image_list,coord_list

file_comp = []
label_list = ['Alnus', 'Alternaria', 'Betula','Carpinus', 'Corylus', 'Fungus', 'Junk', 'Pinus','Poacea', 'Varia']
test_path = r"C:\Users\mishu\OneDrive\Desktop\projects\Image classifications\conv\conv_net\Alternaria_20200819"
list_of_files = os.listdir(test_path)
path = r"C:\Users\mishu\OneDrive\Desktop\projects\Image classifications\conv\conv_net\models"
model_list = os.listdir(path)
models = []
for file_name in model_list:
    file_path = os.path.join(path,file_name)
    models.append(load_model(file_path))

image_stack = []
for file_name in list_of_files[1:]:
    print(file_name)
    # if "compl_P_LZ_--_11572020-08-10 160000.png" != file_name:
    #     continue
    if "ini" in file_name:
        continue
    results = [[] for i in range(len(models))]
    file_path = os.path.join(test_path,file_name)
    image = plt.imread(file_path)
    crop_list,image_list, coord_list= segment_image_2(image)
    image = plt.imread(file_path)
    if len(image_list) ==0:
        continue
    image_stack = np.stack(image_list,axis=0)        
    for i,model in enumerate(models):
        results[i].append( model.predict(image_stack))
    final_res = np.mean(results,axis=0)
    final_res = final_res[0]
    final_res_argmax = np.argmax(final_res,axis=1)
    for i in range(len(image_list)):
        if final_res_argmax[i] == 1 :
            new_image = np.zeros((360,360,4))
            im_shape = crop_list[i].shape
            new_image[int(180-im_shape[0]/2):int(180+im_shape[0]/2),int(180-im_shape[1]/2):int(180+im_shape[1]/2),0] = crop_list[i]
            new_image[int(180-im_shape[0]/2):int(180+im_shape[0]/2),int(180-im_shape[1]/2):int(180+im_shape[1]/2),1] = crop_list[i]
            new_image[int(180-im_shape[0]/2):int(180+im_shape[0]/2),int(180-im_shape[1]/2):int(180+im_shape[1]/2),2] = crop_list[i]
            new_image[int(180-im_shape[0]/2):int(180+im_shape[0]/2),int(180-im_shape[1]/2):int(180+im_shape[1]/2),3] = crop_list[i] * 0. + 1

            path_save = r"C:\Users\mishu\OneDrive\Desktop\projects\Image classifications\conv\conv_net\New folder"
            plt.imsave(os.path.join(path_save,file_name.replace(".png", label_list[final_res_argmax[i]]+str(i)+".png")),new_image)
            plt.close()

    plt.figure(figsize=(20,20))
    plt.imshow(image,cmap ="gray")
    for i in range(len(image_list)):
        if final_res_argmax[i] != 1000 :  
            rect = patches.Rectangle((coord_list[i][2],coord_list[i][0]),
                                    coord_list[i][3]-coord_list[i][2],
                                    coord_list[i][1]-coord_list[i][0],
                                    linewidth=1,edgecolor='r',facecolor='none')
            ax = plt.gca()
            ax.add_patch(rect)
            plt.text(coord_list[i][3],
                    coord_list[i][1],
                    str(i)+label_list[final_res_argmax[i]]+str(np.round(final_res[i][final_res_argmax[i]],decimals=2)),
                    fontsize=20,color="red")
    path_save_segment = r"C:\Users\mishu\OneDrive\Desktop\projects\Image classifications\conv\conv_net\New folder"
    
    plt.savefig(os.path.join(path_save_segment,file_name.replace(".png","classified1.png")))
    plt.close()
   



