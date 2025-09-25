# -*- coding: utf-8 -*-
"""
Created on Mon Aug 23 14:58:01 2021

@author: Sanjana
"""
import os 
Date='y20m10d07'
sess_num='2'
Session='Session'+str(sess_num)
Plant='Plant2'
root='D:/Plant Water Stress/Data/Trial_005/Patches/Plant2/'+str(Date)+'/'+str(Session)+'/'

def delete_images(start_index,stop_index):
    for i in range(int(start_index),int(stop_index)+1):
        src_path=root+f'{i:04}'+'.jpg'
        if os.path.exists(src_path):
            os.remove(src_path)
            print("removing:",f'{i:04}'+'.jpg')

#Image index from which yel is no longer visible because camera has shifted too much right
start_index='0520'
#The last image index in the given Session
stop_index='0958'
delete_images(start_index,stop_index)


last_index=int(start_index)-1
#Vertcal scan steps (40 for Trial005)
step=40
#For the first set of vertical scans, image index from which yel has disappeared since camera has moved up.
s1=33
s2=step-1
for i in range(0,last_index,40):
    start=i+s1
    stop=i+s2
    start_index=f'{start:04}'
    #print('start=',start_index)
    stop_index=f'{stop:04}'
    #print('stop=',stop_index)
    delete_images(start_index,stop_index)
    