
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[17]:


import shutil
import os
import os.path
from os import listdir
from os.path import isfile, join
from PIL import Image
import pathlib


# In[18]:


cwd = os.getcwd()


# In[19]:


photo_dir = cwd + '/photos'
meta_json = cwd + '/meta/json'



# In[20]:


meta_json_files = [f for f in listdir(meta_json) if isfile(join(meta_json, f))]


# In[21]:


meta_json_files


# In[22]:


meta_json_files.remove('.DS_Store')


# In[23]:


meta_json_files


# In[24]:


all_ids = []


# In[25]:


for file in meta_json_files:
 
    
    print('-'*10)
    
    print(file)
    
    rr = pd.read_json(cwd+ '/meta/json/' +file)
    print(rr.shape)
    photo_id = list(rr['photo']) 
    
    
    print('-'*10)
    all_ids.extend(photo_id)




# In[26]:


new_image_ids = []


# In[28]:


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
    
    




# In[31]:


len(new_image_ids)


# In[32]:


with open('photos/photos.txt') as f:
    lines = f.readlines()


# In[33]:


#split the the index from url or the first 9 digits

img_id = []
img_url = []
for i in range(len(lines)):
    img_id.append(lines[i].split(',')[0])
    img_url.append(lines[i].split(',')[1])


# In[34]:


img_id


# In[35]:


img_url


# In[36]:


new_image_ids


# In[37]:


for idx in range(len(new_image_ids)):
            new_image_ids[idx] = str(new_image_ids[idx]).zfill(9)


# In[38]:


new_image_ids


# In[39]:


new_photo_txt = []
for i in range(len(img_id)):
    
    if img_id[i] in new_image_ids:
        
        new_photo_txt.append(str(img_id[i]) + ',' + str(img_url[i]))
        
 


# In[42]:


new_photo_txt = ''.join(str(e) for e in new_photo_txt)


# In[43]:


text_file = open("photos.txt", "w")
text_file.write(new_photo_txt)
text_file.close()

