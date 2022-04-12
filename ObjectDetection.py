# -*- coding: utf-8 -*-
"""
Created on Fri Apr  8 15:44:04 2022

@author: U7DCHENN
"""

import json
import numpy as np
import pandas as pd
import os
import cv2
import matplotlib.pyplot as plt


import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F

import random
from sklearn.model_selection import train_test_split

path = r'C:\Users\U7DCHENN\Downloads\cv-programming-task-candidate-develop\images'
class_dict = {'wallet:0'}

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Blue color in BGR 
color = (255, 0, 0) 

# Line thickness of 3 pixels 
thickness = 3

'''
    read data from the json file 
    return data frame with bb coordinates
'''
def read_labelled_data(path):
    table = []
    for file in os.listdir(path):
        if file.endswith(".json"):
            f =  open((os.path.join(path, file)), "r") 
            data= json.load(f)
           
            for d in data:
                image = d['image']
                image_path = path + f'\{image}'
                #print(image_path)
                label = d['annotations'][0]['label']
                
                width = d['annotations'][0]['coordinates']['width']
                height = d['annotations'][0]['coordinates']['height']
                
                w = width / 2
                h = height /2 
                
                x = int(d['annotations'][0]['coordinates']['x'])
                x_min = x - w
                x_max = x_min + width 
                
                y = int(d['annotations'][0]['coordinates']['y'])
                
                y_min = y - h
                y_max = y_min + height 
               
                table.append([image_path, label, width, height, x_min, y_min, x_max, y_max])
                
    df = pd.DataFrame(table, columns=['image_path', 'label', 'width', 'height', 'x_min', 'y_min', 'x_max', 'y_max'])               
    return df

'''
    convert the label to categorical variable
'''        
df  = read_labelled_data(path)
df['label'] = df['label'].astype('category')
df.head(5)
len(df)

'''
    Creates a mask for the bounding box of same shape as image
    assumes 0 as background nonzero object
'''

def bb_to_mask(bb, x):
    rows,cols,_ = x.shape
    M = np.zeros((rows, cols))
    bb = bb.astype(np.int)
    M[bb[0]:bb[2], bb[1]:bb[3]] = 1.
    return M

"""
    Creates bounding box to  mask  
"""
def mask_to_bb(M):
    
    cols, rows = np.nonzero(M)
    #print("Rows:", rows)
    if len(cols)==0: 
        return np.zeros(4, dtype=np.float32)
    top_row = np.min(rows)
    left_col = np.min(cols)
    bottom_row = np.max(rows)
    right_col = np.max(cols)
    #print(top_row, bottom_row)
    return np.array([left_col, top_row, right_col, bottom_row], dtype=np.float32)

'''
     create bounding box array from the data frame
'''
def create_bb_array(x):
    #print("coord:", np.array([x[5],x[4],x[7],x[6]]))
    return np.array([x[5],x[4],x[7],x[6]])

"""
     Resize an image and its bounding box 
     write new image to new path
     
"""
def resize_image_bb(read_path,write_path,bb,sz):
   
    #print('read_path', read_path)
    im = read_image(read_path)
    #print(im.shape)
    im_resized = cv2.resize(im, (sz, sz))
    #print(im_resized.shape)
    Y_resized = cv2.resize(bb_to_mask(bb, im), (sz, sz))
    filename = os.path.basename(read_path)
    new_path = os.path.join(write_path, filename)
    #print(new_path)
    cv2.imwrite(new_path, cv2.cvtColor(im_resized, cv2.COLOR_RGB2BGR))
    return  new_path, mask_to_bb(Y_resized)

def read_image(path):
    #print("read_image:", path)
    return (cv2.imread(path, cv2.COLOR_BGR2RGB))
    

#add  new paths and bounding boxes to data frame

train_path_resized = r'C:\Users\U7DCHENN\Downloads\cv-programming-task-candidate-develop\images_resized'
def updated_df(df):
    new_paths = []
    new_bbs = []
    for index, row in df.iterrows():
        #print(" Row Values:", row.values)
        new_path, new_bb = resize_image_bb(row['image_path'], train_path_resized, create_bb_array(row.values),300)
        #print(new_path)
        new_paths.append(new_path)
        new_bbs.append(new_bb)
    df['new_path'] = new_paths
    df['new_bb'] = new_bbs
    return df

df_train = updated_df(df)

#Visualize image
def visualize(df_train):
    im = cv2.imread(df_train.values[5][8])
    plt.imshow(im)
    x = df_train.values[5]
    bb = df_train.values[5][9]
    #masked image
    Y = bb_to_mask(bb, im)

    start_point = (int(bb[1]), int(bb[0]))
    end_point = (int(bb[3]), int(bb[2]))

    bb_img = cv2.rectangle(im, start_point, end_point, color, thickness) 
    
    plt.figure()
    
    fig, ax = plt.subplots(1,2)
    #ax[0].imshow(im)
    ax[0].imshow(Y, cmap='gray')
    ax[1].imshow(bb_img)

    plt.show()


visualize(df_train)


'''Data Augmentation '''
'''
    crop the images
'''
def crop(im, row, col, target_r, target_c): 
    return im[row:row+target_r, col:col+target_c]

'''
    center crop the image
'''
def center_crop(x, r_pix=8):
    row, col,*_ = x.shape
    c_pix = round(r_pix*col/row)
    return crop(x, r_pix, c_pix, row-2*r_pix, col-2*c_pix)

""" 
    Rotates an image by deg degrees
"""
def rotate_cv(im, deg, y=False, mode=cv2.BORDER_REFLECT, interpolation=cv2.INTER_AREA):
    
    rows,cols,*_ = im.shape
    M = cv2.getRotationMatrix2D((cols/2,rows/2),deg,1)
    if y:
        return cv2.warpAffine(im, M,(cols,rows), borderMode=cv2.BORDER_CONSTANT)
    return cv2.warpAffine(im,M,(cols,rows), borderMode=mode, flags=cv2.WARP_FILL_OUTLIERS+interpolation)

""" 
    Returns a random crop of masked image
"""
def random_cropXY(x, Y, r_pix=8):
    
    rows, cols,_ = x.shape
    c_pix = round(r_pix*cols/rows)
    rand_r = random.uniform(0, 1)
    rand_c = random.uniform(0, 1)
    start_r = np.floor(2*rand_r*r_pix).astype(int)
    start_c = np.floor(2*rand_c*c_pix).astype(int)
    xx = crop(x, start_r, start_c, rows-2*r_pix, cols-2*c_pix)
    YY = crop(Y, start_r, start_c, rows-2*r_pix, cols-2*c_pix)
    return xx, YY

def transformsXY(path, bb, transforms):
    x = cv2.imread(path).astype(np.float32)
    x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)/255
    Y = bb_to_mask(bb, x)
    if transforms:
        # rotation of the wallet
    
        rdeg = (np.random.random()-.50)*20
        x = rotate_cv(x, rdeg)
        Y = rotate_cv(Y, rdeg, y=True)
        #Randmly cropped wallet
        x, Y = random_cropXY(x, Y)
    else:
        x, Y = center_crop(x), center_crop(Y)
    return x, mask_to_bb(Y)

def create_corner_rect(bb, color='red'):
    bb = np.array(bb, dtype=np.float32)
    return plt.Rectangle((bb[1], bb[0]), bb[3]-bb[1], bb[2]-bb[0], color=color,
                         fill=False, lw=3)

def show_corner_bb(im, bb):
    plt.imshow(im)
    plt.gca().add_patch(create_corner_rect(bb))
    
im = cv2.imread(df.values[5][8])
plt.imshow(im) 

im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
show_corner_bb(im, df.values[5][9])

im, bb = transformsXY(df.values[5][8], df.values[5][9], True)
show_corner_bb(im, bb)

#################Split the dataset################
'''
    Split the dataset
'''
df_train = df_train.reset_index()

X = df_train [['new_path', 'new_bb']]
y = df_train['label']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)


class WalletDataset(Dataset):
    def __init__(self, paths, bb, y, transforms=False):
        self.transforms = transforms
        self.paths = paths.values
        self.bb = bb.values
        self.y = y.values
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, idx):
        path = self.paths[idx]
        y_class = self.y[idx]
        y_bb = self.bb[idx]
        x, y_bb = transformsXY(path, self.bb[idx], self.transforms)

        return x, y_class, y_bb

train_ds = WalletDataset(X_train['new_path'],X_train['new_bb'] ,y_train, transforms=False)
valid_ds = WalletDataset(X_val['new_path'],X_val['new_bb'],y_val)

batch_size = 32
train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
valid_dl = DataLoader(valid_ds, batch_size=batch_size)



