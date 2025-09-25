# -*- coding: utf-8 -*-
"""
Created on Sun Jul 11 13:56:09 2021

@author: Sanjana
"""
from distutils.dir_util import copy_tree
import os
import cv2
import matplotlib.pyplot as plt
import glob
from sklearn.feature_extraction import image
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.manifold import TSNE
import scipy

import numpy as np

# copy subdirectory example
trial_num = '005'
#Count number of folders in chosen trial
root='D:/Plant Water Stress/Data/Trial_'+str(trial_num)+'/Trial'+trial_num
list=os.listdir(root)
print("There are "+ str(len(list))+" days in Trial" + trial_num)
print("Following are the contents of the chosen Trial")
list.sort()
print(list)
#date_num=input("Enter date from the list shown above in the same format:")
#print("You have selected Date:",(date_num))

for date_num in list:
    
    src="D:/Plant Water Stress/Data/Trial_005/Sessions/"+str(date_num)+'/'
    min_b = []
    max_b = []

    min_g = []
    max_g = []

    min_r = []
    max_r = []
    
    for session in os.listdir(src):
        fromDirectory = src+str(session)+'/'
        toDirectory = "D:/Plant Water Stress/Data/Trial_005/NormalizedImages/"+str(date_num)+'/'+str(session)+'/'
        if not os.path.exists(toDirectory):
            print("Creating:",toDirectory)
            os.makedirs(toDirectory)
        copy_tree(fromDirectory, toDirectory)
        
        min_b = []
        max_b = []

        min_g = []
        max_g = []

        min_r = []
        max_r = []
        
        directory = toDirectory+'*.*'
        print("Normalizing images from "+str(session)+'of day'+str(date_num))
        for filename in glob.glob(directory):
            im1 = cv2.imread(filename)
            b = im1[:,:,0]                                                              # Blue Channel
            g = im1[:,:,1]                                                              # Green Channel
            r = im1[:,:,2]                                                              # NIR Channel
            min_b.append(np.min(b))                                                     # Minimum Pixel Value of Blue Channel
            max_b.append(np.max(b))                                                     # Maximum Pixel Value of Blue Channel
            min_g.append(np.min(g))                                                     # Minimum Pixel Value of Green Channel
            max_g.append(np.max(g))                                                     # Maximum Pixel Value of Green Channel
            min_r.append(np.min(r))                                                     # Minimum Pixel Value of NIR Channel
            max_r.append(np.max(r))                                                     # Maximum Pixel Value of NIR Channel
        max_b = np.mean(max_b)                                                          # Mean of Maximum Pixel value of Blue Channel
        min_b = np.mean(min_b)                                                          # Mean of Minimum Pixel value of Blue Channel

        max_g = np.mean(max_g)                                                          # Mean of Maximum Pixel value of Green Channel
        min_g = np.mean(min_g)                                                          # Mean of Minimum Pixel value of Green Channel

        max_r = np.mean(max_r)                                                          # Mean of Maximum Pixel value of Red Channel
        min_r = np.mean(min_r)                                                          # Mean of Minimum Pixel value of Red Channel

        for filename in glob.glob(directory):
            print("Normalizing",filename)
            im1 = cv2.imread(filename)
            b = im1[:,:,0]
            g = im1[:,:,1]
            r = im1[:,:,2]

            new_b = 255 * (b - min_b)/(max_b - min_b)                                   # Normalized pixel for Blue Channel
            new_g = 255 * (g - min_g)/(max_g - min_g)                                   # Normalized pixel for Green Channel
            new_r = 255 * (r - min_r)/(max_r - min_r)                                   # Normalized pixel for NIR Channel
            cv2.imwrite(filename,cv2.merge((new_b,new_g,new_r)))
    


