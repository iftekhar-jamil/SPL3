# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 15:48:33 2019

@author: User
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt


lower_yellow = np.array([10,150,200])
upper_yellow = np.array([80,250,255])

lower_blue = np.array([100,90,150]) 
upper_blue = np.array([145,255,255]) 

lower_red = np.array([150,150,150]) 
upper_red = np.array([255,255,255]) 


def isInRange(a,b,c):
    if(a>=lower_yellow[0] and a<=upper_yellow[0]):
        if(b>=lower_yellow[1] and b<=upper_yellow[1]):
            if(c<=lower_yellow[2] and c<=upper_yellow[2]):
                return 0
    if(a>=lower_red[0] and a<=upper_red[0]):
        if(b>=lower_red[1] and b<=upper_red[1]):
            if(c<=lower_red[2] and c<=upper_red[2]):
                return 1
    if(a>=lower_blue[0] and a<=upper_blue[0]):
        if(b>=lower_blue[1] and b<=upper_blue[1]):
            if(c<=lower_blue[2] and c<=upper_blue[2]):
                return 2        
     
    return -1


inputImage = cv2.imread('F:/Projects/Python Project/final.png')
hsv = cv2.cvtColor(inputImage, cv2.COLOR_BGR2HSV)
#plt.imshow(inputImage)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
mask_red =  cv2.inRange(hsv, lower_red, upper_red)
mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)   

inputImageGray = cv2.cvtColor(inputImage, cv2.COLOR_BGR2GRAY)
seg = 1
height, width, channels = inputImage.shape


for j in range (50,width-1,50):
  print(j)  
  red=0
  blue=0
  yellow=0
  for k in range (j,j+50):
        for i in range (0,height):
             if(k>1350):
                break
#             a,b,c =  inputImage[i,k];
#            if(a==0 and b==0 and c==0):
#                continue;
#            if(isInRange(a,b,c)==0):
#                y = y+1
#            elif(isInRange(a,b,c)==1):
#                r = r+1
#            else:
#                b = b+1
             if(mask_red[i,k]>0):
                 red = red+1
             if(mask_blue[i,k]>0):
                 blue = blue+1    
             if(mask_yellow[i,k]>0):
                 yellow = yellow+1
                 
  if(yellow>red and yellow>blue):
      print("Y")
  if(red>yellow and red>blue):
      print("R")
  if(blue>red and blue>yellow):
      print("B")    
  seg = seg+1
        