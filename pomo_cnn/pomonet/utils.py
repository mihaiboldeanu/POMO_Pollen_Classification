# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 07:45:38 2021

@author: Mihai Boldeanu
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os
import hashlib
import gc

import skimage.color
import skimage.filters
import skimage.io
# import skimage.viewer
import skimage.measure
import skimage.color
from skimage.morphology import erosion, closing, disk

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.metrics import categorical_crossentropy,sparse_categorical_crossentropy
from tensorflow.keras.losses import binary_crossentropy
from scipy.ndimage import rotate

from tensorflow.keras.preprocessing.image import load_img
def validate_coords(coord_list,expanded_mask,alternaria):
    coord_list_valid = []
    for coordinate in coord_list:
        new_img =  np.zeros((1,960,1280,1))
        corners = coordinate
        
        area = alternaria[:,corners[0]:corners[1],corners[2]:corners[3],:]

        if small(coordinate,area):
            idx_small_particle = np.where(expanded_mask[:,corners[0]:corners[1],corners[2]:corners[3],:]==2)
            expanded_mask[:,corners[0]:corners[1],corners[2]:corners[3],:][idx_small_particle] = 1
            continue
        coord_list_valid.append(coordinate)
    return coord_list_valid, expanded_mask

def dice_coef(y_true, y_pred):
    nb_classes = K.int_shape(y_pred)[-1]
    y_pred1 = K.reshape(y_pred, (-1, nb_classes))
    y_true1 = tf.cast(K.reshape(y_true, (-1, 1))[:,0],tf.int32)
    y_true1 = K.one_hot(y_true1, nb_classes)
    y_true1 = y_true1[:,1:]
    y_pred1 = y_pred1[:,1:]
    
    y_true_f = K.flatten(y_true1)
    y_pred_f = K.flatten(y_pred1)
    

    intersection = y_true_f * y_pred_f
    score = 2. * K.sum(intersection) / (K.sum(y_true_f) + K.sum(y_pred_f))
    return score
def iou_coef(y_true, y_pred):
    smooth = 0.0001
    nb_classes = K.int_shape(y_pred)[-1]
    y_pred1 = K.reshape(y_pred, (-1, nb_classes))
    y_true1 = tf.cast(K.reshape(y_true, (-1, 1))[:,0],tf.int32)
    y_true1 = K.one_hot(y_true1, nb_classes)
    y_true1 = y_true1[:,1:]
    y_pred1 = y_pred1[:,1:]
    
    y_true_f = K.flatten(y_true1)
    y_pred_f = K.flatten(y_pred1)
    intersection = y_true_f * y_pred_f
    score = (K.sum(intersection) + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) - K.sum(intersection) + smooth)
    return score

def dice_loss(y_true, y_pred):
    smooth = 0.0001
    nb_classes = K.int_shape(y_pred)[-1]
    y_pred1 = K.reshape(y_pred, (-1, nb_classes))
    y_true1 = tf.cast(K.reshape(y_true, (-1, 1))[:,0],tf.int32)
    y_true1 = K.one_hot(y_true1, nb_classes)
    # y_true1 = y_true1[:,1:]
    # y_pred1 = y_pred1[:,1:]
    
    y_true_f = K.flatten(y_true1)
    y_pred_f = K.flatten(y_pred1)
    intersection = y_true_f * y_pred_f
    score = (2. * K.sum(intersection) + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return 1. - score


def iou_loss(y_true, y_pred):
    smooth = 0.0001
    nb_classes = K.int_shape(y_pred)[-1]
    y_pred1 = K.reshape(y_pred, (-1, nb_classes))
    y_true1 = tf.cast(K.reshape(y_true, (-1, 1))[:,0],tf.int32)
    y_true1 = K.one_hot(y_true1, nb_classes)
    # y_true1 = y_true1[:,1:]
    # y_pred1 = y_pred1[:,1:]
    
    y_true_f = K.flatten(y_true1)
    y_pred_f = K.flatten(y_pred1)
    intersection = y_true_f * y_pred_f
    score = (K.sum(intersection) + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) - K.sum(intersection) + smooth)
    return 1. - score

def bce_dice_loss(y_true, y_pred):
    return sparse_categorical_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)

def bce_dice_iou_loss(y_true, y_pred):
    return sparse_categorical_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred) + iou_loss(y_true, y_pred)

def bce_logdice_loss(y_true, y_pred):
    return sparse_categorical_crossentropy(y_true, y_pred) - K.log(1. - dice_loss(y_true, y_pred))

def macro_IOU(y_true, y_pred):
    nb_classes = K.int_shape(y_pred)[-1]
    y_pred = K.reshape(y_pred, (-1, nb_classes))
    
    y_true = tf.cast(K.reshape(y_true, (-1, 1))[:,0],tf.int32)
    y_true = K.one_hot(y_true, nb_classes)
    
    true_pixels = K.argmax(y_true, axis=-1) # exclude background
    pred_pixels = K.argmax(y_pred, axis=-1)
    
    iou = []
    flag = tf.convert_to_tensor(-1, dtype='float64')
    for i in range(nb_classes):
        
        true_labels = K.equal(true_pixels, i)
        pred_labels = K.equal(pred_pixels, i)
        
        inter = tf.cast(true_labels & pred_labels,tf.int32)
        union = tf.cast(true_labels | pred_labels,tf.int32)
        
        cond = (K.sum(union) > 0) & (K.sum(tf.cast(true_labels,tf.int32)) > 0)
        res = tf.cond(cond, lambda: K.sum(inter)/K.sum(union), lambda: flag)
        iou.append(res)
    iou = tf.stack(iou)
    legal_labels = tf.greater(iou, flag)
    iou = tf.gather(iou, indices=tf.where(legal_labels))
    return K.mean(iou) 

def mean_IOU_ignoring_first_label(y_true, y_pred):
    '''
    IOU loss to minimize. Pass to model as loss during compile statement
    '''
    nb_classes = K.int_shape(y_pred)[-1]
    y_pred = K.reshape(y_pred, (-1, nb_classes))
    
    y_true = tf.cast(K.reshape(y_true, (-1, 1))[:,0],tf.int32)
    y_true = K.one_hot(y_true, nb_classes)
    
    
    pred_pixels = K.argmax(y_pred, axis=-1)
    y_pred = K.one_hot(pred_pixels, nb_classes)
    
    legal_labels = ~tf.cast(y_true[:,0], tf.bool)
    y_pred = tf.gather(y_pred, indices=tf.where(legal_labels))
    y_true = tf.gather(y_true, indices=tf.where(legal_labels))
    
    
    inter = y_true * y_pred
    union = K.sum(y_true + y_pred - inter,axis=-1)
    inter = K.sum(inter,axis=-1)
    
    
    iou = inter/union
    
    return iou

def sparse_accuracy_ignoring_first_label(y_true, y_pred):
    nb_classes = K.int_shape(y_pred)[-1]
    y_pred = K.reshape(y_pred, (-1, nb_classes))

    y_true = K.one_hot(tf.cast(K.flatten(y_true),tf.int32), nb_classes)
    unpacked = tf.unstack(y_true, axis=-1)
    legal_labels = ~tf.cast(unpacked[0], tf.bool)
    y_true = tf.stack(unpacked[1:], axis=-1)
    true_positives = K.equal(K.argmax(y_true, axis=-1), K.argmax(y_pred, axis=-1))
    good_vals = K.sum(tf.cast(legal_labels & true_positives, tf.float32) )
    all_vals = K.sum(tf.cast(legal_labels, tf.float32))
    acc = good_vals / all_vals

    return acc


def segment_image_2(array):
    
    image = array[0,:,:,0]
    to_big = 0
    blur = skimage.filters.gaussian(image, sigma=0.7)### Slight blur to help image segmentation
    mask = blur > 0.1
    # plt.figure(figsize=(10,12))
    # plt.imshow(mask,cmap="viridis")
    # plt.show()
    mask = closing(mask,selem=disk(2))
    labeled_image = skimage.measure.label(mask, connectivity=1., return_num=True)### Actual segmentation CCA
    
    values_mask = labeled_image[0]### Get a list of segments
    # plt.figure(figsize=(10,12))
    # plt.imshow(values_mask,cmap="viridis")
    # plt.show()
    # plt.figure(figsize=(10,12))
    # plt.imshow(image,cmap="gray")
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
        
        if im_shape[0]>350 or im_shape[1]>350:
            to_big +=1
            continue
        if im_shape[0]<20 or im_shape[1]<20 or len(np.where(values_mask==uniq)[0]) < 100:
            image_modified[x_min:x_max,y_min:y_max] = 0
            continue
        
        new_image = np.zeros((360,360,1))
        new_image[int(180-im_shape[0]/2):int(180+im_shape[0]/2),int(180-im_shape[1]/2):int(180+im_shape[1]/2),0] = crop_im
        image_modified[x_min:x_max,y_min:y_max] = 0
        image_list.append(new_image)
        crop_list.append(crop_im)
        coord_list.append((x_min,x_max,y_min,y_max))
    
    # plt.figure(figsize=(10,12))
    # plt.imshow(image_modified,cmap="gray")
    # plt.show()
    if np.sum(image_modified)==0:
        return crop_list,image_list,coord_list,to_big
    else:
        image_modified_2 = np.copy(image_modified)
        blur = skimage.filters.gaussian(image_modified_2, sigma=1.5)### Slight blur to help image segmentation
        
        blur_mask = np.copy(image_modified_2)
        blur_mask[np.where(blur_mask==0 )] = 140/255. 
        mask = blur_mask < np.mean(blur[np.where(blur>0)])
        mask = erosion(mask,selem=disk(2))

        labeled_image = skimage.measure.label(mask, connectivity=1., return_num=True)### Actual segmentation CCA
        values_mask = labeled_image[0]
        
        # plt.figure(figsize=(10,12))
        # plt.imshow(values_mask,cmap="viridis")
        # plt.show()
        # plt.figure(figsize=(10,12))
        # plt.imshow(image_modified_2,cmap="gray")
        # plt.show()
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
                to_big +=1
                continue
            if im_shape[0]<30 or im_shape[1]<30 or len(np.where(values_mask==uniq)[0]) < 100:
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

        return crop_list,image_list,coord_list,to_big
    
def segment_image(array):
    
    image = array[0,:,:,0]
    blur = skimage.filters.gaussian(image, sigma=0.7)### Slight blur to help image segmentation
    mask = blur > 0.1

    mask = closing(mask,selem=disk(2))
    #mask = opening(mask,selem=disk(2))
    labeled_image = skimage.measure.label(mask, connectivity=1., return_num=True)### Actual segmentation CCA
    
    values_mask = labeled_image[0]### Get a list of segments
    # plt.figure()
    # plt.imshow(values_mask)
    # plt.show()
    coord_list = []

    uniques = np.unique(values_mask)
    for uniq in uniques:
        image_temp = np.zeros(image.shape)
        indexex = np.where(values_mask==uniq)
        image_temp[indexex] = image[indexex]

        x_min = np.min(indexex[0])-2
        x_max = np.max(indexex[0])+2
        y_min = np.min(indexex[1])-2
        y_max = np.max(indexex[1])+2
        if x_max - x_min >=960  or y_max - y_min >=960:
            continue
        if y_max>=960:
            y_max = 959
        if x_max >= 1280:
            x_max = 1279
        coord_list.append((x_min,x_max,y_min,y_max))
    
    
    return coord_list

def small(coordinate,area):
    x_min,x_max,y_min,y_max = coordinate
    x_size =  x_max - x_min
    y_size =  y_max - y_min
    
    if x_size<25 or y_size<25:
        return True
    if np.where(area)[0].size <100:
        return True
    return False

def big(coordinate,area):
    x_min,x_max,y_min,y_max = coordinate
    x_size =  x_max - x_min
    y_size =  y_max - y_min
    
    if x_size>360 or y_size>360:
        return True

    return False
    

def augment(img,flipud,fliplr,rotate_angle):
    temp_x = np.array(img)
    if rotate_angle:
        
            temp_x = np.around(rotate(temp_x,rotate_angle,reshape=False))
            temp_x[np.where(temp_x<0)] = 0

    if flipud:
        temp_x = np.flip(temp_x,axis=0)
    if fliplr:
        temp_x = np.flip(temp_x,axis=1)
        
    return temp_x


def expand_mask(mask):
    expanded_mask = np.argmax(mask, axis=-1)
    expanded_mask = np.expand_dims(expanded_mask, axis=-1)
    return expanded_mask

def plot_original_mask(original,expanded_mask,name):

    fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(20,10))

    ax1.imshow(original[0,:,:,0],cmap="gray")
    ax1.set_xticks([])
    ax1.set_yticks([])

    ax2.imshow(np.around(expanded_mask[0,:,:]),vmin=0,vmax=2)
    ax2.set_xticks([])
    ax2.set_yticks([])

    plt.tight_layout()
    plt.savefig(name+".png")
    plt.draw()
    ax1.cla()
    ax2.cla()
    fig.clf()
    plt.close(fig)
    plt.close("all")
    gc.collect()
    
def save_image(x_unet,expanded_mask,file_path):
    f_path,f_name = os.path.split(file_path)

    if (expanded_mask==2).any():
        f_name = "classified_alternaria_"+f_name
    else:
        f_name = "classified_junk_"+f_name

    name = os.path.join(f_path,f_name)  
    plot_original_mask(x_unet,expanded_mask,name)
    
def get_one_class(array):
    binary_array = array == 2
    return binary_array
def check_corners(corners):
    new_corners = list(corners)
    if corners[1] >= 960:
        new_corners[1] = 959
    
    if corners[3] >= 1280:
        new_corners[3] = 1279
    return corners

def get_crop(image,corners):
    new_image = np.zeros((360,360,1))
    
    im_shape = [corners[1]-corners[0],corners[3]-corners[2]]
    crop = np.array(image)[corners[0]:corners[1],corners[2]:corners[3]]
    if corners[1]-corners[0] < 360 and corners[3]-corners[2]:
    
        new_image[int(180-im_shape[0]/2):int(180+im_shape[0]/2),
                  int(180-im_shape[1]/2):int(180+im_shape[1]/2),0] = crop
    else:
        im_shape = [np.minimum(360,corners[1]-corners[0]),
                    np.minimum(360,corners[3]-corners[2])]
        
        new_image[int(180-im_shape[0]/2):int(180+im_shape[0]/2),
                  int(180-im_shape[1]/2):int(180+im_shape[1]/2),0] = crop[0:im_shape[0],0:im_shape[1]]
        
    return new_image


def hash_function(file_path):
    file = file_path # Location of the file (can be set a different way)
    BLOCK_SIZE = 65536 # The size of each read from the file
    
    file_hash = hashlib.sha256() # Create the hash object, can use something other than `.sha256()` if you wish
    with open(file, 'rb') as f: # Open the file to read it's bytes
        fb = f.read(BLOCK_SIZE) # Read from the file. Take in the amount declared above
        while len(fb) > 0: # While there is still data being read from the file
            file_hash.update(fb) # Update the hash
            fb = f.read(BLOCK_SIZE) # Read the next block from the file
    
    return (file_hash.hexdigest()) # Get the hexadecimal digest of the hash

def read_meta_data(file_path):
    ret_list = []
    with open(file_path,"r") as f:
        for line in f:
            ret_list.append(line.split(";"))
    return ret_list
img_size = (960, 1280)

def get_unique_pictures(path):

    data_files = sorted([os.path.join(path, fname) for fname in os.listdir(path) if fname.endswith(".png")])

    unique_list = []
    data_dict = {}
    for idx,file_name in enumerate(data_files):
        file_path = os.path.join(path,file_name)
        
        h = hash_function(file_path)
        
        temp_dict = {}
        temp_dict['file'] = file_name

        coords = [0,0,0,0]
        temp_dict['coords'] = coords#metadata[idx+1][4]
        if h in unique_list:
            data_dict[h].append(temp_dict)
            continue
        else:
            data_dict[h] = [temp_dict]
    
            unique_list.append(h)
    return data_dict
def plot_image_with_particles(file_path,coord_list):
    img_size = (1280, 960)
    img = load_img(file_path, 
                   target_size=img_size,
                   color_mode="grayscale")
    temp_x = augment(img,0,0,0)
    fig,ax = plt.subplots()
    ax.imshow(temp_x,"gray")
    for altceva_coords in coord_list:
        y_min,x_min,y_offset,x_offset = altceva_coords
        y_min = round(y_min * 0.75)
        x_min = round(x_min * 1.3333333333333333)
        y_offset = round(y_offset * 0.75)
        x_offset = round(x_offset * 1.3333333333333333)
        y_max = y_min + y_offset
        x_max = x_min + x_offset
        
            
        rect = patches.Rectangle((y_min, x_min),
                                 y_offset,
                                 x_offset,
                                 linewidth=1,
                                 edgecolor='r',
                                 facecolor='none')

        # Add the patch to the Axes
        ax.add_patch(rect)
        
def extract_crops_of_images(file_path,coord_list):
    img_size = (1280, 960)
    img = load_img(file_path, 
                   target_size=img_size,
                   color_mode="grayscale")
    temp_x = augment(img,0,0,0)
    ret_list = []
    for altceva_coords in coord_list:
        y_min,x_min,y_offset,x_offset = altceva_coords
        y_min = round(y_min * 0.75)
        x_min = round(x_min * 1.3333333333333333)
        y_offset = round(y_offset * 0.75)
        x_offset = round(x_offset * 1.3333333333333333)
        y_max = y_min + y_offset
        x_max = x_min + x_offset
        patch = temp_x[x_min:x_max,y_min:y_max]
        im_shape = patch.shape
        new_image = np.zeros((360,360,1))
        new_image[int(180-im_shape[0]/2):int(180+im_shape[0]/2),
                  int(180-im_shape[1]/2):int(180+im_shape[1]/2),0] = patch
        ret_list.append( new_image)
    return ret_list

def crop_of_images(file_path,coord_list):
    img_size = (1280, 960)
    img = load_img(file_path, 
                   target_size=img_size,
                   color_mode="grayscale")
    temp_x = augment(img,0,0,0)
    ret_list = []
    for altceva_coords in coord_list:
        y_min,x_min,y_offset,x_offset = altceva_coords
        y_min = round(y_min * 0.75)
        x_min = round(x_min * 1.3333333333333333)
        y_offset = round(y_offset * 0.75)
        x_offset = round(x_offset * 1.3333333333333333)
        y_max = y_min + y_offset
        x_max = x_min + x_offset
        patch = np.copy(temp_x)
        patch_mask = np.zeros((1280, 960))
        patch_mask[x_min:x_max,y_min:y_max] = 1
        ret_list.append( patch * patch_mask)
    return ret_list