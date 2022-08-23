# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 07:45:38 2021

@author: Mihai Boldeanu
"""


import os
import random

import numpy as np

from skimage.morphology import closing,erosion, disk
from scipy.ndimage import rotate

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import load_img




class Pollen_mixed(keras.utils.Sequence):
    """Helper to iterate over the data (as Numpy arrays)."""

    def __init__(self, batch_size,
                 img_size,
                 input_img_paths,
                 target_img_paths,
                 cropped_img_paths,
                 value_dict,
                 augment=True,
                 junk_value=1,
                 step_per_epoch=2000,
                 background_path=None,
                 validation=False,
                 extra_particle=None):
        self.len = step_per_epoch
        self.batch_size = batch_size
        self.p_real = Pollen_v2 (batch_size,
                                 img_size,
                                 input_img_paths,
                                 target_img_paths,
                                 augment=augment,
                                 junk_value=junk_value,)
        self.p_synth = Pollen_synthetic(batch_size,
                                        img_size,
                                        cropped_img_paths,
                                        value_dict,
                                        step_per_epoch=step_per_epoch,
                                        background_path=background_path,
                                        validation=validation,
                                        extra_particle=extra_particle)


    def __len__(self):
        
        return self.len
    def __getitem__(self, idx):
        """Returns tuple (input, target) correspond to batch #idx."""
        
        i = idx * self.batch_size
        if i > (self.len - 5):
            return self.p_real.__getitem__(i)
        
        if np.random.random(1) < 0.3:
            # print("real" +str(i))
            return self.p_real.__getitem__(i)
        else:
            # print("synth"+str(i))
            return self.p_synth.__getitem__(i)
        
        
        
    def on_epoch_end(self):
        self.p_real.on_epoch_end()
   

class Pollen_v2(keras.utils.Sequence):
    """Helper to iterate over the data (as Numpy arrays)."""

    def __init__(self, batch_size, img_size, input_img_paths,
                 target_img_paths,augment=True,junk_value=1):
        
        self.batch_size = batch_size
        self.img_size = img_size
        self.input_img_paths = input_img_paths
        self.target_img_paths = target_img_paths
        self.augment = augment
        self.junk_value = junk_value

    def __len__(self):
        # 
        # self.on_epoch_end()
        return len(self.target_img_paths) // self.batch_size

    def __getitem__(self, idx):
        """Returns tuple (input, target) correspond to batch #idx."""
        
        i = idx * self.batch_size
        batch_input_img_paths = self.input_img_paths[i : i + self.batch_size]
        batch_target_img_paths = self.target_img_paths[i : i + self.batch_size]


        x, y, w = self.__data_generation(batch_input_img_paths,
                                         batch_target_img_paths)
       
        
        return x, y, w
    def __data_generation(self, batch_input_path,
                                batch_target_img_paths):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        x = np.zeros((self.batch_size,self.img_size[0],self.img_size[1],1))
        y = np.zeros((self.batch_size,self.img_size[0],self.img_size[1],1))
        w = np.zeros((self.batch_size,self.img_size[0],self.img_size[1],1))
        
        for i, _ in enumerate(batch_input_path):
            # print(batch_input_path[i])
            img = load_img(batch_input_path[i], target_size=self.img_size,color_mode="grayscale")
            mask = load_img(batch_target_img_paths[i], target_size=self.img_size, color_mode="grayscale")

            flipud, fliplr, rotate_angle  = 0, 0 ,0
            wavy_hori = False
            wavy_vert = False
            a=0
            W=0
            
        
            if self.augment:
                
                flipud = np.random.random(1) > 0.75
                fliplr = np.random.random(1) > 0.75
                wavy_vert = np.random.random(1) > 0.85
                wavy_hori = np.random.random(1) > 0.85
                a = np.random.randint(3,8)
                W = np.random.randint(3,8)
            
                if np.random.random(1) > 0.75:
                    rotate_angle = np.random.randint(0,360,1)[0]
                    #rotate_angle = 0
                else:
                    rotate_angle = 0
            
            temp_x = self.augment_position(img,flipud,fliplr,wavy_vert,wavy_hori,rotate_angle,A=a,w=W)
            temp_y = self.augment_position(mask,flipud,fliplr,wavy_vert,wavy_hori,rotate_angle,A=a,w=W)
            if self.augment:
                temp_x = self.augment_color(temp_x)
            

            
            x[i,:,:,0] = np.clip(temp_x,0,255)
            y[i,:,:,0] = temp_y
           
        w += 0.1
        w[np.where(y>0)]=1
        w[np.where(y>1)]=2
        
        return tf.convert_to_tensor(x/255.), tf.convert_to_tensor(y), tf.convert_to_tensor(w)

    def augment_position(self,img,flipud,fliplr,wavy_vert,wavy_hori,rotate_angle,A=8,w=6):
        temp_x = np.array(img)
        max_value = np.max(temp_x)
        if rotate_angle:

                temp_x = np.around(rotate(temp_x,rotate_angle,reshape=False))
                temp_x[np.where(temp_x<0)] = 0

                
        if flipud:
            temp_x = np.flip(temp_x,axis=0)
        if fliplr:
            temp_x = np.flip(temp_x,axis=1)
            
        ### Deformation  augmentation ###    
        if wavy_hori:
            temp_x = self.make_wavy(temp_x,A=A,w=w)
        if wavy_vert:
            temp_x = self.make_wavy(temp_x,A=A,w=w,horizontal=False)
            
        temp_x = np.clip(temp_x,0,max_value)     
        return temp_x
    
    def augment_color(self,img):
        temp_x = np.array(img)
         ### Histogram shift augmentation ###
        if random.random()  > 0.85:
            temp_x = self.make_log(temp_x)  
        if random.random()  > 0.85:
            temp_x = self.make_square(temp_x)  
        if random.random()  > 0.85:        
            temp_x = self.make_square_root(temp_x)    
            
        
        return temp_x
    def make_wavy(self,image,A=4,w=6,horizontal=True):
        
        A = A
        w = w / image.shape[1]
        shift = lambda x: A * np.sin(2.0*np.pi* x * w)
        
        image_waves = np.zeros(image.shape)

        if horizontal:
            for i in range(image.shape[0]):
                image_waves[i,:] = np.roll(image[i,:], int(shift(i)))
        else:
            for i in range(image.shape[1]):
                image_waves[:,i] = np.roll(image[:,i], int(shift(i)))
        return image_waves
    
    def make_square(self,image):
        m = np.max(image)
        temp_x = image**2.
        if m != 0:
            temp_x = temp_x/np.max(temp_x) * m 
            
        return temp_x
    
    def make_square_root(self,image):
        m = np.max(image)
        temp_x = np.sqrt(image)
        if m != 0:
            temp_x = temp_x/np.max(temp_x) * m  
             
        return temp_x
    def make_log(self,image):
        m = np.max(image)
        temp_x = np.log(image+1.)
        if m != 0:
            temp_x = temp_x/np.max(temp_x) * m  
             
        return temp_x
    
    def make_bright(self,image):
        brightness = 0.3
        alpha= 1.0 + random.uniform(-brightness, brightness)
        temp_x = image * alpha
        temp_x = np.clip(temp_x,0,1)
        return temp_x
    def on_epoch_end(self):
        # if self.augment:
        #     print("len was called")
        seed = np.random.randint(129345)
        random.Random(seed).shuffle(self.input_img_paths)
        random.Random(seed).shuffle(self.target_img_paths)       
    

    
class Pollen_synthetic(keras.utils.Sequence):
    """Helper to iterate over the data (as Numpy arrays)."""

    def __init__(self,
                 batch_size,
                 img_size,
                 input_img_paths,
                 value_dict,
                 step_per_epoch=10000,
                 background_path=None,
                 validation=False,
                 extra_particle=None):
        
        self.batch_size = batch_size
        self.step_per_epoch = step_per_epoch
        self.img_size = img_size
        self.input_img_paths = input_img_paths
        self.value_dict = value_dict
        self.background_path = background_path
        self.validation = validation
        self.extra_particle = extra_particle

    def __len__(self):
        return self.step_per_epoch

    def __getitem__(self, idx):
        """Returns tuple (input, target) correspond to batch #idx."""

        # Generate data
        if self.validation:
            idx_seed = idx
        else:
            idx_seed = np.random.randint(0,913829128)
            
        random.seed(idx_seed)  
        # np.random.seed(idx)
        x, y, w = self.__data_generation(idx_seed)
        
        return x, y, w
    
    def __data_generation(self, idx):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        i = 0
        x = np.zeros((self.batch_size,self.img_size[0],self.img_size[1],1))
        y = np.zeros((self.batch_size,self.img_size[0],self.img_size[1],1))
        w = np.zeros((self.batch_size,self.img_size[0],self.img_size[1],1))
        # x = []
        # y = []
        # w = []
        if self.validation:
            random.seed(idx)
        
        if self.background_path:
            paths = [os.path.join(self.background_path,file_name) for file_name in os.listdir(self.background_path)]
        while i < self.batch_size:
            
            part = random.randint(20,48)
            
            image = np.zeros((1320,1640))
            mask =  np.zeros((1320,1640))

            if self.background_path and random.random() > 0.9 and not self.validation:
                
                back_path = random.choice(paths)
                background = load_img(back_path, target_size=(960,1280),color_mode="grayscale")
                background = np.array(background)
                flipud = random.random() > 0.75
                fliplr = random.random() > 0.75
                if random.random() > 0.75:
                    rotate_angle = random.randint(0,360)
                else:
                    rotate_angle = 0
                background = self.augment(background,flipud,fliplr,rotate_angle)
                background_mask = background > 0
                background_mask = closing(background_mask)
                background_mask = background_mask * self.value_dict["junk"]
                image[180:1320-180,180:1640-180] += background
                mask[180:1320-180,180:1640-180] += background_mask
                part = random.randint(8,20)


            selection = self.get_pollens(self.input_img_paths,
                                         number_examples=part,
                                         extra_particle=self.extra_particle,
                                         seed=idx)
            
            image,mask = self.add_pollen(image,mask,
                                         selection,self.value_dict,seed=idx)
                
            image = image[180:1320-180,180:1640-180]
            mask = mask[180:1320-180,180:1640-180]
            # images = self.split_image(image, (256,256))
            # masks = self.split_image(mask, (256,256))
            x[i,:,:,0] = np.clip(image,0,255)
            y[i,:,:,0] = mask
            
            # y[i,:,:,0] = mask
            
            i+=1
        # x = np.array(x)   
        # y = np.array(y)
        # w = y * 0
        w += 0.1
        w[np.where(y==2)]=2
        w[np.where(y==1)]=1
        #w[np.where(y==11)]=0.5

        return tf.convert_to_tensor(x/255.), tf.convert_to_tensor(y), tf.convert_to_tensor(w)
     
    def get_pollens(self,pollen_dict,extra_particle=None,
                    number_examples = 10,
                    seed=10):
        if self.validation:
            random.seed(seed)
        
        keys = list(pollen_dict.keys())
        ret_particles = []
        
        while len(ret_particles) < number_examples:
            key = random.choice(keys,)
            ret_particles.append([key,random.choice(pollen_dict[key])])
            
        if extra_particle:    
            for i in range(np.random.randint(1,3)):
                ret_particles.append([extra_particle,random.choice(pollen_dict[extra_particle])])# Force to have at least one alternaria particle
       
        return ret_particles
    

    def add_pollen(self,current_image,current_mask,particles,value_dict,seed=10):
        if self.validation:
            random.seed(seed)
        # np.random.seed(seed)
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
                flipud = random.random() > 0.5
                fliplr = random.random() > 0.5
                if random.random() > 0.75:
                    rotate_angle = random.randint(0,360)
                else:
                    rotate_angle = 0
                img = self.augment(img,flipud,fliplr,rotate_angle)
            mask = ( img > 0 )
            mask = erosion(mask)
            reverse_mask = np.logical_not(mask)
            value_mask = mask * value_dict[key]
            current_image[x_min:x_max,y_min:y_max] = current_image[x_min:x_max,y_min:y_max] * reverse_mask + mask*img
            if key == "alternaria":
                
                value_mask =  erosion(mask,selem=disk(3)) * value_dict[key]
                background_mas = np.logical_xor( mask ,erosion(mask,selem=disk(3))) * value_dict["junk"]
                current_mask[x_min:x_max,y_min:y_max] = current_mask[x_min:x_max,y_min:y_max] * reverse_mask + value_mask + background_mas
            else:
                
                current_mask[x_min:x_max,y_min:y_max] = current_mask[x_min:x_max,y_min:y_max] * reverse_mask + value_mask

        return current_image, current_mask
    
    def augment(self,img,flipud,fliplr,rotate_angle):
        temp_x = np.array(img)
        ### Positional augmentation ###
        if rotate_angle:
                temp_x = np.around(rotate(temp_x,rotate_angle,reshape=False))
                temp_x[np.where(temp_x<0)] = 0
        if flipud:
            temp_x = np.flip(temp_x,axis=0)
        if fliplr:
            temp_x = np.flip(temp_x,axis=1)
            
        if img.shape[0] == 360:
            
            ### Histogram shift augmentation ###
            if random.random()  > .85:
                temp_x = self.make_bright(temp_x)  
                
            elif random.random()  > .85:
                temp_x = self.make_square(temp_x)  
                
            elif random.random()  > .85:
                temp_x = self.make_log(temp_x)  
                
            elif random.random()  > .85: 
                temp_x = self.make_square_root(temp_x)    
                
            ### Deformation and cropping augmentation ###    
            if random.random() > .90:
                temp_x = self.make_hole(temp_x)
            if random.random() > .8:
                temp_x = self.make_wavy(temp_x)
            if random.random() > .8:
                temp_x = self.make_wavy(temp_x,horizontal=False)
                
        temp_x = np.clip(temp_x,0,255)    
        return temp_x
    
    def make_hole(self,image):
        shape_dims = image.shape
        
        x = random.randint(int(shape_dims[0]/3), int(2*shape_dims[0]/3))
        width = random.randint(int(shape_dims[0]/40.), int(shape_dims[0]/10.))
        y = random.randint(int(shape_dims[1]/3), int(2*shape_dims[1]/3))
        height = random.randint(int(shape_dims[1]/40.), int(shape_dims[1]/10.))
        
        image[x:x+width,y:y+height] = random.random() *255.
        
        return image
    def make_log(self,image):
        m = np.max(image)
        temp_x = np.log(image+1.)
        if m != 0:
            temp_x = temp_x/np.max(temp_x) * m  
             
        return temp_x
    def make_wavy(self,image,horizontal=True):
        
        A = random.randint(3,6)
        w = random.randint(3,6) / image.shape[1]
        shift = lambda x: A * np.sin(2.0*np.pi* x * w)
        image_waves = np.zeros(image.shape)
        if horizontal:
            for i in range(image.shape[1]):
                image_waves[i,:] = np.roll(image[i,:], int(shift(i)))
        else:
            for i in range(image.shape[0]):
                image_waves[:,i] = np.roll(image[:,i], int(shift(i)))
        return image_waves
    
    def make_square(self,image):
        m = np.max(image)
        temp_x = image**2.
        if m != 0:
            temp_x = temp_x/np.max(temp_x) * m 
            
        return temp_x
    
    def make_square_root(self,image):
        m = np.max(image)
        temp_x = np.sqrt(image)
        if m != 0:
            temp_x = temp_x/np.max(temp_x) * m  
             
        return temp_x
    
    def make_bright(self,image):
        # brightness = 0.3
        alpha= 1.0 + random.uniform(-0.5, 0.25)
        temp_x = image * alpha
        temp_x = np.clip(temp_x,0,255)
        return temp_x
        
            
        

    def split_segments(self,array_size,window_size):
    
        full_iterations = array_size//window_size
        
        difference = array_size - full_iterations * window_size
        if difference == 0:
            splits = full_iterations
            splits = [(i*window_size,(i+1)* window_size )  for i in range(full_iterations)]
        else:
            
            s = int((array_size - window_size)/(full_iterations))
            # full_iterations +=1
            splits = []
            difference_part = full_iterations * window_size - array_size 
            difference_part = difference_part / full_iterations

    
            for i in range(0,array_size,s):
                if i+window_size > array_size:
                    break
                splits.append([i ,
                               i+window_size])
    
        return splits
    
    def split_image(self,image,new_shape):
        image_stack = []
        
        original_image_shape = image.shape
        
        dim_x = original_image_shape[0]
        dim_x_new = new_shape[0]
        dim_y = original_image_shape[1]
        dim_y_new = new_shape[1]
        
        x_splits = self.split_segments(dim_x,dim_x_new)
        y_splits = self.split_segments(dim_y,dim_y_new)
        
        combinations = [[x,y] for x in x_splits for y in y_splits ]
        
        for combination in combinations:
            small_image = image[combination[0][0]:combination[0][1],
                                combination[1][0]:combination[1][1]] 
            image_stack.append(small_image)

        return np.array(image_stack)
    
    
    def combine_image(self,image_stack,old_shape,new_shape):
        image = np.zeros(old_shape)
        dim_x = old_shape[0]
        dim_x_new = new_shape[0]
        dim_y = old_shape[1]
        dim_y_new = new_shape[1]
        
        x_splits = self.split_segments(dim_x,dim_x_new)
        y_splits = self.split_segments(dim_y,dim_y_new)
        
        combinations = [[x,y] for x in x_splits for y in y_splits ]
        for i,combination in enumerate(combinations):
            image[combination[0][0]:combination[0][1],combination[1][0]:combination[1][1]] = image_stack[i]+ i*10/255.
            
        return image