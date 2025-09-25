# -*- coding: utf-8 -*-
"""
Created on Sun Jul 11 13:59:14 2021

@author: Sanjana
"""
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import shutil


src_path='D:/Plant Water Stress/Data/Trial_005/Trial005/'
list=os.listdir(src_path)
print("There are "+ str(len(list))+" days in Trial005")
print("Following are the contents of the chosen Trial")
list.sort()
print(list)

for date_num in list:

    #date_num=input("Enter date from the list shown above in the same format:")
    
    print("Now going through Date:",date_num)
    src=src_path+str(date_num)+'/'
    list1_n=[]
    list1=os.listdir(src)
    for i in list1:
        t=i[15:23]
        list1_n.append(t)
    unique_list=[]
    for x in list1_n:
        # check if exists in unique_list or not
        if x not in unique_list:
            unique_list.append(x)
    unique_list.sort()
    print("Following Timestamps are available for this date(format: hr_min_sec) :",unique_list)


    dst='D:/Plant Water Stress/Data/Trial_005/Sessions/'+str(date_num)+'/'
    if not os.path.exists(dst):
        print("Creating date subfolder:",dst)
        os.makedirs(dst)

    for j in range(len(unique_list)):
        dst_path=dst+'Session'+str(j+1)+'/'
        if not os.path.exists(dst_path):
            print("Creating session subfolder:",dst_path)
            os.makedirs(dst_path)
        t_stamp=unique_list[j]
        for item in list1:
            if t_stamp in item:
                print("Copying file:"+str(item)+'to Session'+str(j+1))
                shutil.copyfile(src+str(item), dst_path+str(item))
 