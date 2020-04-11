from __future__ import print_function
from scipy import misc
import os
import numpy as np
import random
import sys
from PIL import Image
from itertools import chain

def load_test_data(phone, dataset_dir, test_size, IMAGE_SIZE, PATCH_SIZE, target):
    test_directory_lq = dataset_dir + str(phone) + "/test_patches/"
    test_directory_hq = dataset_dir + target + "/test_patches/"
    
    if(test_size <= len([name for name in os.listdir(test_directory_lq)
                           if os.path.isfile(os.path.join(test_directory_lq, name))])):
        NUM_TEST_IMAGES = test_size
    else:
        NUM_TEST_IMAGES = len([name for name in os.listdir(test_directory_lq)
                           if os.path.isfile(os.path.join(test_directory_lq, name))])

    test_data = np.zeros((NUM_TEST_IMAGES, PATCH_SIZE, PATCH_SIZE, 3))
    test_target = np.zeros((NUM_TEST_IMAGES, PATCH_SIZE, PATCH_SIZE, 3))

    for i in range(0, NUM_TEST_IMAGES):
        if(phone == "Nova2i" or phone == "iPhone8"):
            I = np.asarray(Image.open(test_directory_lq + "(" + str(i) + ').jpg').crop((0, 0, PATCH_SIZE, PATCH_SIZE)))
        else:
            I = np.asarray(Image.open(test_directory_lq + str(i) + '.jpg').crop((0, 0, PATCH_SIZE, PATCH_SIZE)))
        I = np.float32(I)/255
        test_data[i, :] = I
        if(phone == "Nova2i" or phone == "iPhone8"):
            I = np.asarray(Image.open(test_directory_hq + "(" + str(i) + ').jpg').crop((0, 0, PATCH_SIZE, PATCH_SIZE)))
        else:
            I = np.asarray(Image.open(test_directory_hq + str(i) + '.jpg').crop((0, 0, PATCH_SIZE, PATCH_SIZE)))
        I = np.float32(I)/255
        test_target[i, :] = I

        if i % 100 == 0:
            print(str(round(i * 100 / NUM_TEST_IMAGES)) + "% done", end="\r")

    return test_data, test_target


def load_train_data(phone, dataset_dir, TRAIN_SIZE, IMAGE_SIZE, PATCH_SIZE, target, data_idx, augment=False, train_data_size=-1):
    
    train_directory_lq = dataset_dir + str(phone) + "/train_patches/"
    train_directory_hq = dataset_dir + target + "/train_patches/"

    NUM_TRAINING_IMAGES = len([name for name in os.listdir(train_directory_lq)
                               if os.path.isfile(os.path.join(train_directory_lq, name))])
    if(train_data_size < NUM_TRAINING_IMAGES and train_data_size!=-1):
        NUM_TRAINING_IMAGES = train_data_size
        
    NUM_HQ_IMAGES = len([name for name in os.listdir(train_directory_lq)])
    NUM_LQ_IMAGES = len([name for name in os.listdir(train_directory_lq)])
    reload = True
    augment_size = 1
    
    if(augment):
        augment_size = 5

    # if TRAIN_SIZE == -1 then load all images
    if((NUM_TRAINING_IMAGES <= TRAIN_SIZE) or TRAIN_SIZE == -1):
        diff_en = TRAIN_SIZE - NUM_TRAINING_IMAGES
        if(diff_en > 0):
            TRAIN_IMAGES = chain(range(0,NUM_TRAINING_IMAGES), range(0,diff_en))
        else:
            TRAIN_IMAGES = range(0,NUM_TRAINING_IMAGES)
        
        
            
        reload = False
    else:
        be = int(data_idx * TRAIN_SIZE)
        en = int((data_idx+1) * TRAIN_SIZE)
        TRAIN_IMAGES = range(be,en)
        
        if(en > NUM_TRAINING_IMAGES):
            en = NUM_TRAINING_IMAGES
            TRAIN_IMAGES = range(be,en)
    TRAIN_IMAGES = list(TRAIN_IMAGES)
    random.shuffle(TRAIN_IMAGES)
    train_data = np.zeros((len(TRAIN_IMAGES), PATCH_SIZE, PATCH_SIZE, 3))
    train_target = np.zeros((len(TRAIN_IMAGES), PATCH_SIZE, PATCH_SIZE, 3))
    

    i = 0
    for img in TRAIN_IMAGES:
        
        if(phone == "sony" or phone == "canon"):
            I = Image.open(train_directory_lq + str(img) + '.jpg').crop((0, 0, PATCH_SIZE, PATCH_SIZE))
        else:
            I = Image.open(train_directory_lq + '(' + str(img) + ').jpg').crop((0, 0, PATCH_SIZE, PATCH_SIZE))
        I_save = np.asarray(I)    
        I_save = np.float32(I_save) / 255
        train_data[i, :] = I_save

        if(target == "canon" or target == "sony"):
            I = Image.open(train_directory_hq + str(img) + '.jpg').crop((0, 0, PATCH_SIZE, PATCH_SIZE))
        else:
            I = Image.open(train_directory_hq + '(' + str(img) + ').jpg').crop((0, 0, PATCH_SIZE, PATCH_SIZE))
        I_save = np.asarray(I)    
        I_save = np.float32(I_save) / 255
        train_target[i, :] = I_save

        i += augment_size
        if i % 100 == 0:
            print(str(round(i * 100 / TRAIN_SIZE)) + "% done", end="\r")
    print(i)
    return train_data, train_target, reload
