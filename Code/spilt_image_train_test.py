
# coding: utf-8


import pandas as pd
import numpy as np



import shutil
import os
import os.path
from os import listdir
from os.path import isfile, join
import pandas as pd
from PIL import Image
import pathlib



cwd = os.getcwd()




meta_json = cwd + '/meta/json'



from os import listdir
from os.path import isfile, join
meta_json_files = [f for f in listdir(meta_json) if isfile(join(meta_json, f))]




listdir(meta_json)


meta_json_files.remove('.DS_Store')



image_dir = cwd + '/images'




image_files = [f for f in listdir(image_dir) if isfile(join(image_dir, f))]



len(image_files)


# Get all the images in each category



img_name_List = []
img_ext_List = []


for i in image_files:
    img_name = i.split('.',1)[0]
    img_ext = i.split('.',1)[1]
 
    img_name_List.append(img_name)
    
    img_ext_List.append(img_ext)
    


rr = pd.read_json(cwd+ '/meta/json/retrieval_outerwear.json')

photo_id = list(rr['photo']) 



for file in meta_json_files:
    sub_folder_name_filename = file.split('_',1)[-1]
    #print(sub_folder_name_filename)
    sub_folder_name = sub_folder_name_filename.split('.')[0] #name of the folder
    #print(sub_folder_name)
        
    rr = pd.read_json(cwd+ '/meta/json/' +file)
        
    photo_id = list(rr['photo']) 
    
    for idx in range(len(photo_id)):
        photo_id[idx] = str(photo_id[idx]).zfill(9)
    
    
    
    
    all_imgs = []
    #all_ext = []
    for img_idx in photo_id:
        
        if img_idx in img_name_List:
            
            index = img_name_List.index(img_idx)
            
      
            
            all_imgs.append(img_idx + '.' + img_ext_List[index])
    
    TRAIN_SET = all_imgs[len(all_imgs)//3:]
    
    TEST_SET = all_imgs[:len(all_imgs)//3]
    
    
    
    
    #  TRAIN SET        
    for img_data in TRAIN_SET:
        destination_folder = str(cwd) + '/Dataset/Train/' + str(sub_folder_name) # + str(img_data)


        if not os.path.exists(destination_folder):
            os.makedirs(destination_folder)
            
            
        q = pathlib.Path(destination_folder + '/' + str(img_data))
        
        shutil.copy(image_dir + '/' + img_data , q)
                
                

                
    #  TEST SET        
    for img_data in TEST_SET:
        destination_folder = str(cwd) + '/Dataset/Test/' + str(sub_folder_name) # + str(img_data)


        if not os.path.exists(destination_folder):
            os.makedirs(destination_folder)
            
            
        q = pathlib.Path(destination_folder + '/' + str(img_data))
        
        shutil.copy(image_dir + '/' + img_data , q)
                
 

