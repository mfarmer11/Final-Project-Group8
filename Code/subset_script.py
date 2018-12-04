
# coding: utf-8


import pandas as pd
import numpy as np



import shutil
import os
import os.path
from os import listdir
from os.path import isfile, join
from PIL import Image
import pathlib


cwd = os.getcwd()



photo_dir = cwd + '/photos'
meta_json = cwd + '/meta/json'




meta_json_files = [f for f in listdir(meta_json) if isfile(join(meta_json, f))]



meta_json_files.remove('.DS_Store')



all_ids = []


for file in meta_json_files:
 
    
    print('-'*10)
    
    print(file)
    
    rr = pd.read_json(cwd+ '/meta/json/' +file)
    print(rr.shape)
    photo_id = list(rr['photo']) 
    
    
    print('-'*10)
    all_ids.extend(photo_id)



new_image_ids = []




for file in meta_json_files:
 
   
    print(file)
    
    meta_data = pd.read_json(cwd+ '/meta/json/' +file)    
    #Reduce images tot 15,000
    print(meta_data.shape)


    new_sample_data = meta_data.sample(n = 15000)
    print(new_sample_data.shape)
    photo_id = list(new_sample_data['photo']) 
    
    #extend the new image ids to all the New IMAGE IDs
    
    new_image_ids.extend(photo_id)
    
    

len(new_image_ids)



with open('photos/photos.txt') as f:
    lines = f.readlines()



#split the the index from url or the first 9 digits

img_id = []
img_url = []
for i in range(len(lines)):
    img_id.append(lines[i].split(',')[0])
    img_url.append(lines[i].split(',')[1])



for idx in range(len(new_image_ids)):
            new_image_ids[idx] = str(new_image_ids[idx]).zfill(9)



new_photo_txt = []
for i in range(len(img_id)):
    
    if img_id[i] in new_image_ids:
        
        new_photo_txt.append(str(img_id[i]) + ',' + str(img_url[i]))
        
 


new_photo_txt = ''.join(str(e) for e in new_photo_txt)


text_file = open("photos.txt", "w")
text_file.write(new_photo_txt)
text_file.close()

