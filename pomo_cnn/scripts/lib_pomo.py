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
import random
import gc

import skimage.color
import skimage.filters
import skimage.io
import skimage.viewer
import skimage.measure
import skimage.color
from skimage.morphology import erosion, dilation, opening
from skimage.morphology import closing, white_tophat,black_tophat,disk

from scipy.ndimage import rotate

from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.models import Sequential,load_model,Model
from tensorflow.keras.layers import Conv2D,Dropout,Dense,MaxPooling2D,Activation,Conv2DTranspose,GlobalAveragePooling2D
from tensorflow.keras.layers import Input,Flatten,BatchNormalization,Add,add,SeparableConv2D,UpSampling2D
from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping,ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam,SGD
from tensorflow.keras import regularizers
import tensorflow as tf
from tensorflow.keras.utils import plot_model
from tensorflow import keras


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
    
    if x_size<33 or y_size<33:
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
    
    new_image[int(180-im_shape[0]/2):int(180+im_shape[0]/2),int(180-im_shape[1]/2):int(180+im_shape[1]/2),0] = np.array(image)[corners[0]:corners[1],corners[2]:corners[3]]
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
    metadata_file_path = os.path.join(path,"meta-data.csv")
    if os.path.exists(metadata_file_path):
        metadata = read_meta_data(metadata_file_path)
        data_files = [m[0] for m in metadata[1:]]
        coords_flag = True
    else:
        coords_flag = False
        data_files = sorted([os.path.join(path, fname) for fname in os.listdir(path) if fname.endswith(".png")])

    unique_list = []
    data_dict = {}
    for idx,file_name in enumerate(data_files):
        file_path = os.path.join(path,file_name)
        
        h = hash_function(file_path)
        
        temp_dict = {}
        temp_dict['file'] = file_name
        if coords_flag:
            coords = metadata[idx+1][4].split(',')
            coords = [int(c) for c in coords]
        else:
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
def get_model(img_size, num_classes):
    inputs = Input(shape=img_size )

    ### [First half of the network: downsampling inputs] ###

    # Entry block
    x = Conv2D(32, 3, strides=2, padding="same",use_bias=False,kernel_initializer='glorot_normal')(inputs)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    previous_block_activation = x  # Set aside residual

    # Blocks 1, 2, 3 are identical apart from the feature depth.
    for filters in [32, 64, 128,256]:
        x = Activation("relu")(x)
        x = Conv2D(filters, 3, padding="same",use_bias=False,kernel_initializer='glorot_normal')(x)
        x = BatchNormalization()(x)

        x = Activation("relu")(x)
        x = Conv2D(filters, 3, padding="same",use_bias=False,kernel_initializer='glorot_normal')(x)
        x = BatchNormalization()(x)

        x = MaxPooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual = Conv2D(filters, 1, strides=2, padding="same",kernel_initializer='glorot_normal')(
            previous_block_activation
        )
        x = add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    ### [Second half of the network: upsampling inputs] ###

    for filters in [256,128, 64, 32, 32]:
        x = Activation("relu")(x)
        x = Conv2DTranspose(filters, 3, padding="same",use_bias=False,kernel_initializer='glorot_normal')(x)
        x = BatchNormalization()(x)

        x = Activation("relu")(x)
        x = Conv2DTranspose(filters, 3, padding="same",use_bias=False,kernel_initializer='glorot_normal')(x)
        x = BatchNormalization()(x)

        x = UpSampling2D(2)(x)

        # Project residual
        residual = UpSampling2D(2)(previous_block_activation)
        residual = Conv2D(filters, 1, padding="same",kernel_initializer='glorot_normal')(residual)
        x = add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    # Add a per-pixel classification layer
    outputs = Conv2D(num_classes, 3, activation="softmax", padding="same")(x)

    # Define the model
    model = Model(inputs, outputs)
    return model

def get_pollens(pollen_dict,number_examples = 10):
    keys = list(pollen_dict.keys())
    ret_particles = []
    while len(ret_particles)<number_examples:
        key = random.choice(keys)
        ret_particles.append([key,random.choice(pollen_dict[key])])
    return ret_particles

def add_pollen(current_image,current_mask,particle,value_dict):
    key, path = particle
    img = load_img(path, 
                   target_size=(360,360),
                   color_mode="grayscale")
    
    img = np.array(img)
    y_min = random.randint(0, 1280)
    y_max = y_min + 360
    x_min = random.randint(0, 960)
    x_max = x_min + 360
    
    new_image = np.zeros((1320,1640))
    new_image[x_min:x_max,y_min:y_max] = img
    mask = (new_image>0)
    value_mask = mask * value_dict[key]
    reverse_mask = np.logical_not(mask)
    return current_image * reverse_mask + new_image, current_mask * reverse_mask + value_mask

class Pollen(keras.utils.Sequence):
    """Helper to iterate over the data (as Numpy arrays)."""

    def __init__(self, batch_size,
                 img_size, input_img_paths,
                 target_img_paths1,target_img_paths2,
                 augment=True,junk_value=1):
        self.batch_size = batch_size
        self.img_size = img_size
        self.input_img_paths = input_img_paths
        self.target_img_paths1 = target_img_paths1
        self.target_img_paths2 = target_img_paths2
        self.augment = augment
        self.junk_value = junk_value

    def __len__(self):
        return len(self.target_img_paths1) // self.batch_size

    def __getitem__(self, idx):
        """Returns tuple (input, target) correspond to batch #idx."""
        
        i = idx * self.batch_size
        batch_input_img_paths = self.input_img_paths[i : i + self.batch_size]
        batch_target_img_paths1 = self.target_img_paths1[i : i + self.batch_size]
        batch_target_img_paths2 = self.target_img_paths2[i : i + self.batch_size]

        x, y = self.__data_generation(batch_input_img_paths,
                                      batch_target_img_paths1,
                                      batch_target_img_paths2)
        
        return x, y
    def __data_generation(self,
                          batch_input_path,
                          batch_target_img_paths1,
                          batch_target_img_paths2):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        x = np.zeros((self.batch_size,self.img_size[0],self.img_size[1],1))
        y = np.zeros((self.batch_size,self.img_size[0],self.img_size[1],1))
        for i, _ in enumerate(batch_input_path):
            img = load_img(batch_input_path[i], target_size=self.img_size,color_mode="grayscale")
            img1 = load_img(batch_target_img_paths1[i], target_size=self.img_size, color_mode="grayscale")
            img2 = load_img(batch_target_img_paths2[i], target_size=self.img_size, color_mode="grayscale")
            flipud, fliplr, rotate_angle  = 0, 0 ,0
        
            if self.augment:
                flipud = np.random.random(1) > 0.5
                fliplr = np.random.random(1) > 0.5
                if np.random.random(1) > 0.5:
                    rotate_angle = np.random.randint(0,360,1)[0]
                else:
                    rotate_angle = 0
            
            temp_x = self.augment_f(img,flipud,fliplr,rotate_angle)
            temp_y1 = self.augment_f(img1,flipud,fliplr,rotate_angle)
            temp_y2 = self.augment_f(img2,flipud,fliplr,rotate_angle)
            
            temp_y1 = temp_y1 > 128
            temp_y2 = temp_y2 > 128

            temp_y = temp_y1 * 2 + temp_y2 * self.junk_value
            x[i,:,:,0] = temp_x
            y[i,:,:,0] = temp_y

        
        return tf.convert_to_tensor(x/255.), tf.convert_to_tensor(y)

    def augment_f(self,img,flipud,fliplr,rotate_angle):
        temp_x = np.array(img)
        if rotate_angle:
                temp_x = np.around(rotate(temp_x,rotate_angle,reshape=False))
                temp_x[np.where(temp_x<0)] = 0
    
        if flipud:
            temp_x = np.flip(temp_x,axis=0)
        if fliplr:
            temp_x = np.flip(temp_x,axis=1)
            
        return temp_x
    
    def on_epoch_end(self):
        seed = np.random.randint(10)
        random.Random(seed).shuffle(self.input_img_paths)
        random.Random(seed).shuffle(self.target_img_paths1)
        random.Random(seed).shuffle(self.target_img_paths2)
        
    
class Pollen_synthetic(keras.utils.Sequence):
    """Helper to iterate over the data (as Numpy arrays)."""

    def __init__(self,
                 batch_size,
                 step_per_epoch,
                 img_size,
                 input_img_paths,
                 value_dict,
                 validation=False):
        self.batch_size = batch_size
        self.step_per_epoch = step_per_epoch
        self.img_size = img_size
        self.input_img_paths = input_img_paths
        self.value_dict = value_dict
        self.validation = validation

    def __len__(self):
        return self.step_per_epoch

    def __getitem__(self, idx):
        """Returns tuple (input, target) correspond to batch #idx."""

        # Generate data
        if self.validation:
            random.seed(idx)
        else:
            random.seed(np.random.randint(0,913829128))
            
        x, y = self.__data_generation(idx)

        return x, y
    
    def __data_generation(self, idx):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        i = 0
        x = np.zeros((self.batch_size,self.img_size[0],self.img_size[1],1))
        y = np.zeros((self.batch_size,self.img_size[0],self.img_size[1],1))
        
        while i < self.batch_size:
            
            part = random.randint(10,32)
            
            image = np.zeros((1320,1640))    
            mask =  np.zeros((1320,1640)) 
            
            selection = self.get_pollens(self.input_img_paths,number_examples = part)
            
            image,mask = self.add_pollen(image,mask,selection,self.value_dict)
                
                
            x[i,:,:,0] = image[180:1320-180,180:1640-180]
            y[i,:,:,0] = mask[180:1320-180,180:1640-180]
            i+=1
        
        return tf.convert_to_tensor(x/255.), tf.convert_to_tensor(y)
     
    def get_pollens(self,pollen_dict,number_examples = 10):
        
        
        keys = list(pollen_dict.keys())
        ret_particles = []
        
        while len(ret_particles) < number_examples:
            key = np.random.choice(keys,)
            ret_particles.append([key,random.choice(pollen_dict[key])])
            
        for i in range( np.random.randint(0,5)):
            ret_particles.append(["alternaria",random.choice(pollen_dict['alternaria'])])# Force to have at least one alternaria particle
       
        return ret_particles
    

    def add_pollen(self,current_image,current_mask,particles,value_dict):

        for idx,particle in enumerate(particles):
            key, path = particle
            
            y_min = random.randint(0, 1280)
            y_max = y_min + 360
            x_min = random.randint(0, 960)
            x_max = x_min + 360
            img = load_img(path, 
                        target_size=(360,360),
                        color_mode="grayscale")
            img = np.array(img)
            if not self.validation: 
                if self.augment:
                    flipud = np.random.random(1) > 0.65
                    fliplr = np.random.random(1) > 0.65
                    if np.random.random(1) > 0.75:
                        rotate_angle = np.random.randint(0,360,1)[0]
                    else:
                        rotate_angle = 0
                img = self.augment(img,flipud,fliplr,rotate_angle)
            mask = ( img > 0 )
            reverse_mask = np.logical_not(mask)
            value_mask = mask * value_dict[key]
            current_image[x_min:x_max,y_min:y_max] = current_image[x_min:x_max,y_min:y_max] * reverse_mask + img
            current_mask[x_min:x_max,y_min:y_max] = current_mask[x_min:x_max,y_min:y_max] * reverse_mask + value_mask

        return current_image, current_mask
    
    def augment(self,img,flipud,fliplr,rotate_angle):
        temp_x = np.array(img)
        if rotate_angle:
                temp_x = np.around(rotate(temp_x,rotate_angle,reshape=False))
                temp_x[np.where(temp_x<0)] = 0
    
        if flipud:
            temp_x = np.flip(temp_x,axis=0)
        if fliplr:
            temp_x = np.flip(temp_x,axis=1)
            
        return temp_x

  

def div_to_float(img):
    image=np.array(img)  
    return image/255.


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
        x_out = Dense(18)(x_dense)
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
