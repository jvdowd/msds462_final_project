#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
from matplotlib import pyplot as plt
import numpy as np
import os
import pandas as pd
import random
from skimage import io
from shutil import copyfile
import sys
import time

import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array

pd.set_option("display.max_rows", 999)


# <b>Credits</b><br>
# Code adapted from Yinghan Xu and Yann Henon
# 
# <b>Xu:</b> https://github.com/RockyXu66/Faster_RCNN_for_Open_Images_Dataset_Keras <br>
# <b>Henon:</b> https://github.com/yhenon/keras-rcnn

# In[2]:


tf.__version__


# In[3]:


# List of files with training data

wd = '/Users/jamesvdowd/Documents/MSDS/462/final_project/'

images_boxable_fname = wd+'train-images-boxable-with-rotation.csv'
annotations_bbox_fname = wd+'train-annotations-bbox.csv'
class_descriptions_fname = wd+'class-descriptions-boxable.csv'


# In[4]:


images_boxable = pd.read_csv(images_boxable_fname)
images_boxable.head()


# In[5]:


annotations_bbox = pd.read_csv(annotations_bbox_fname)
annotations_bbox.head()


# In[6]:


class_descriptions = pd.read_csv(class_descriptions_fname, header=None)
class_descriptions.head()


# In[7]:


def plot_bbox(img_id):
    img_url = images_boxable.loc[images_boxable["ImageID"]==img_id]['OriginalURL'].values[0]
    img = io.imread(img_url)
    height, width, channel = img.shape
  
    print(f"Image: {img.shape}")
    bboxs = annotations_bbox[annotations_bbox['ImageID']==img_id]
    for index, row in bboxs.iterrows():
        xmin = row['XMin']
        xmax = row['XMax']
        ymin = row['YMin']
        ymax = row['YMax']
        xmin = int(xmin*width)
        xmax = int(xmax*width)
        ymin = int(ymin*height)
        ymax = int(ymax*height)
        label_name = row['LabelName']
        class_series = class_descriptions[class_descriptions[0]==label_name]
        class_name = class_series[1].values[0]
        print(f"Coordinates: {xmin,ymin}, {xmax,ymax}")
        cv2.rectangle(img, (xmin,ymin), (xmax,ymax), (255,0,0), 5)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, class_name, (xmin,ymin-10), font, 3, (0,255,0), 5)
        plt.figure(figsize=(15,10))
        plt.title('Image with Bounding Box')
        plt.imshow(img)
        plt.axis("off")
        plt.show()


# In[8]:


least_objects_img_ids = annotations_bbox["ImageID"].value_counts().tail(50).index.values


# In[9]:


for img_id in random.sample(list(least_objects_img_ids), 5):
    try:
        plot_bbox(img_id)
    except:
        print('Image Gone')
        


# In[10]:


annotations_bbox['LabelName'].value_counts()

categories = pd.merge(annotations_bbox['LabelName']
         , class_descriptions
         , how = 'left'
         , left_on = 'LabelName'
         , right_on = 0)

categories[1].value_counts()


# In[11]:


class_descriptions.loc[class_descriptions[1].isin(['Boat'
                                                   , 'Bicycle'
                                                   , 'Car'
                                                   , 'Truck'
                                                   , 'Taxi'])]


# In[12]:


# Find the label_name for Boat, Bicycle, Car, Truck, Taxi classes
boat_df = class_descriptions[class_descriptions[1]=='Boat']
bike_df = class_descriptions[class_descriptions[1]=='Bicycle']
car_df = class_descriptions[class_descriptions[1]=='Car']
truck_df = class_descriptions[class_descriptions[1]=='Truck']
taxi_df = class_descriptions[class_descriptions[1]=='Taxi']

label_boat = boat_df[0].values[0]
label_bike = bike_df[0].values[0]
label_car = car_df[0].values[0]
label_truck = truck_df[0].values[0]
label_taxi = taxi_df[0].values[0]


# In[13]:


bbox_boat = annotations_bbox[annotations_bbox['LabelName']==label_boat]
bbox_bike = annotations_bbox[annotations_bbox['LabelName']==label_bike]
bbox_car = annotations_bbox[annotations_bbox['LabelName']==label_car]
bbox_truck = annotations_bbox[annotations_bbox['LabelName']==label_truck]
bbox_taxi = annotations_bbox[annotations_bbox['LabelName']==label_taxi]


# In[14]:


# Check number of objects in the dataset
print('There are %d boars in the dataset' %(len(bbox_boat)))
print('There are %d bikes in the dataset' %(len(bbox_bike)))
print('There are %d cars in the dataset' %(len(bbox_car)))
print('There are %d trucks in the dataset' %(len(bbox_truck)))
print('There are %d taxis in the dataset' %(len(bbox_taxi)))

iid_boat = bbox_boat['ImageID']
iid_bike = bbox_bike['ImageID']
iid_car = bbox_car['ImageID']
iid_truck = bbox_truck['ImageID']
iid_taxi = bbox_taxi['ImageID']


# In[15]:


iid_boat = np.unique(iid_boat)
iid_bike = np.unique(iid_bike)
iid_car = np.unique(iid_car)
iid_truck = np.unique(iid_truck)
iid_taxi = np.unique(iid_taxi)

print('There are %d images which contain boats' % (len(iid_boat)))
print('There are %d images which contain bikes' % (len(iid_bike)))
print('There are %d images which contain cars' % (len(iid_car)))
print('There are %d images which contain truck' % (len(iid_truck)))
print('There are %d images which contain taxi' % (len(iid_taxi)))


# In[16]:


# here I've chosen only 10 images for speed, change it to your liking
n = 1000

sub_iid_boat = random.sample(list(iid_boat), n)
sub_iid_bike = random.sample(list(iid_bike), n)
sub_iid_car = random.sample(list(iid_car), n)
sub_iid_truck = random.sample(list(iid_truck), n)
sub_iid_taxi = random.sample(list(iid_taxi), n)


# In[17]:


sub_boat_df = images_boxable.loc[images_boxable['ImageID'].isin(sub_iid_boat)]
sub_bike_df = images_boxable.loc[images_boxable['ImageID'].isin(sub_iid_bike)]
sub_car_df = images_boxable.loc[images_boxable['ImageID'].isin(sub_iid_car)]
sub_truck_df = images_boxable.loc[images_boxable['ImageID'].isin(sub_iid_truck)]
sub_taxi_df = images_boxable.loc[images_boxable['ImageID'].isin(sub_iid_taxi)]


# In[18]:


print(sub_boat_df.shape)
print(sub_bike_df.shape)
print(sub_car_df.shape)
print(sub_truck_df.shape)
print(sub_taxi_df.shape)


# In[19]:


sub_boat_dict = sub_boat_df[["ImageID", "OriginalURL"]].set_index('ImageID')["OriginalURL"].to_dict()
sub_bike_dict = sub_bike_df[["ImageID", "OriginalURL"]].set_index('ImageID')["OriginalURL"].to_dict()
sub_car_dict = sub_car_df[["ImageID", "OriginalURL"]].set_index('ImageID')["OriginalURL"].to_dict()
sub_truck_dict = sub_truck_df[["ImageID", "OriginalURL"]].set_index('ImageID')["OriginalURL"].to_dict()
sub_taxi_dict = sub_taxi_df[["ImageID", "OriginalURL"]].set_index('ImageID')["OriginalURL"].to_dict()


# In[20]:


mappings = [sub_boat_dict, sub_bike_dict, sub_car_dict
            , sub_truck_dict, sub_taxi_dict]

print(len(mappings))
print(len(mappings[0]))


# In[21]:


classes = ['Boat', 'Bike', 'Car', 'Truck', 'Taxi']


# In[22]:


# download images
for idx, obj_type in enumerate(classes):
    n_issues = 0
    # create the directory
    if not os.path.exists(obj_type):
        os.mkdir(obj_type)
    for img_id, url in mappings[idx].items():
        try:
            img = io.imread(url)
            saved_path = os.path.join(obj_type, img_id+".jpg")
            io.imsave(saved_path, img)
        except Exception as e:
            n_issues += 1
            print(f"Images Issues: {n_issues}")


# In[23]:


ls Taxi | wc -l


# In[24]:


# save images to train and test directory
train_path = 'train'
test_path = 'test'


# In[25]:


for i in range(len(classes)):
    all_imgs = os.listdir(classes[i])
    all_imgs = [f for f in all_imgs if not f.startswith('.')]
    random.shuffle(all_imgs)
    
    limit = int(n*0.8)

    train_imgs = all_imgs[:limit]
    test_imgs = all_imgs[limit:]
    
    # copy each classes' images to train directory
    for j in range(len(train_imgs)):
        original_path = os.path.join(classes[i], train_imgs[j])
        new_path = os.path.join(train_path, train_imgs[j])
        copyfile(original_path, new_path)
    
    # copy each classes' images to test directory
    for j in range(len(test_imgs)):
        original_path = os.path.join(classes[i], test_imgs[j])
        new_path = os.path.join(test_path, test_imgs[j])
        copyfile(original_path, new_path)
        


# In[26]:


label_names = [label_boat
               , label_bike
               , label_car
               , label_truck
               , label_taxi]

train_df = pd.DataFrame(columns=['FileName'
                                 , 'XMin'
                                 , 'XMax'
                                 , 'YMin'
                                 , 'YMax'
                                 , 'ClassName'])

# Find boxes in each image and put them in a dataframe
train_imgs = os.listdir(train_path)
train_imgs = [name for name in train_imgs if not name.startswith('.')]

for i in range(len(train_imgs)):
    sys.stdout.write('Parse train_imgs ' + str(i) + '; Number of boxes: ' + str(len(train_df)) + '\r')
    sys.stdout.flush()
    img_name = train_imgs[i]
    img_id = img_name[0:16]
    tmp_df = annotations_bbox[annotations_bbox['ImageID']==img_id]
    for index, row in tmp_df.iterrows():
        labelName = row['LabelName']
        for i in range(len(label_names)):
            if labelName == label_names[i]:
                train_df = train_df.append({'FileName': img_name, 
                                            'XMin': row['XMin'], 
                                            'XMax': row['XMax'], 
                                            'YMin': row['YMin'], 
                                            'YMax': row['YMax'], 
                                            'ClassName': classes[i]}, 
                                           ignore_index=True)
                


# In[27]:


train_df.tail(n=25)


# In[28]:


train_df.shape


# In[29]:


train_img_ids = train_df["FileName"].head(n=1).str.split(".").str[0].unique()


# In[30]:


for img_id in train_img_ids:
    plot_bbox(img_id)


# In[31]:


test_df = pd.DataFrame(columns=['FileName'
                                 , 'XMin'
                                 , 'XMax'
                                 , 'YMin'
                                 , 'YMax'
                                 , 'ClassName'])

# Find boxes in each image and put them in a dataframe
test_imgs = os.listdir(test_path)
test_imgs = [name for name in test_imgs if not name.startswith('.')]

for i in range(len(test_imgs)):
    sys.stdout.write('Parse test_imgs ' + str(i) + '; Number of boxes: ' + str(len(test_df)) + '\r')
    sys.stdout.flush()
    img_name = test_imgs[i]
    img_id = img_name[0:16]
    tmp_df = annotations_bbox[annotations_bbox['ImageID']==img_id]
    for index, row in tmp_df.iterrows():
        labelName = row['LabelName']
        for i in range(len(label_names)):
            if labelName == label_names[i]:
                test_df = test_df.append({'FileName': img_name, 
                                            'XMin': row['XMin'], 
                                            'XMax': row['XMax'], 
                                            'YMin': row['YMin'], 
                                            'YMax': row['YMax'], 
                                            'ClassName': classes[i]}, 
                                           ignore_index=True)
                


# In[32]:


train_df.to_csv('train.csv')
test_df.to_csv('test.csv')


# In[33]:


train_df.head()


# In[34]:


train_df = pd.read_csv('train.csv')

# for training
with open("annotation.txt", "w+") as f:
    for idx, row in train_df.iterrows():
        img = cv2.imread('train/' + row['FileName'])
        height, width = img.shape[:2]
        x1 = int(row['XMin'] * width)
        x2 = int(row['XMax'] * width)
        y1 = int(row['YMin'] * height)
        y2 = int(row['YMax'] * height)

        fileName = os.path.join(row['FileName'])
        className = row['ClassName']
        f.write(fileName + ',' + str(x1) + ',' + str(y1) + ',' + str(x2) + ',' + str(y2) + ',' + className + '\n')

    


# In[35]:



# for test
with open("test_annotation.txt", "w+") as f:
    for idx, row in test_df.iterrows():
        sys.stdout.write(str(idx) + '\r')
        sys.stdout.flush()
        img = cv2.imread('test/' + row['FileName'])
        height, width = img.shape[:2]
        x1 = int(row['XMin'] * width)
        x2 = int(row['XMax'] * width)
        y1 = int(row['YMin'] * height)
        y2 = int(row['YMax'] * height)

        fileName = os.path.join(row['FileName'])
        className = row['ClassName']
        f.write(fileName + ',' + str(x1) + ',' + str(y1) + ',' + str(x2) + ',' + str(y2) + ',' + className + '\n')


# In[36]:


print(len(train_df))
print(len(test_df))


# In[ ]:




