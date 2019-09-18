# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 15:48:33 2019

@author: User
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
path = [cv2.imread(file) for file in glob.glob("F:/Projects/Python Project/Images/*.png")]

lower_yellow = np.array([10,150,200])
upper_yellow = np.array([80,250,255])

lower_blue = np.array([100,90,150]) 
upper_blue = np.array([145,255,255]) 

lower_red = np.array([150,150,150]) 
upper_red = np.array([255,255,255]) 
images = []

def findInit(w,h):
    for j in range (0,h):
        for i in range (0,w):
            if(mask_red[j,i]>0 or mask_blue[j,i]>0 or mask_yellow[j,i]>0):
                return j

def findFinish(w,h):
   last = -1 
   for j in range (0,h):
        for i in range (0,w):
            if(mask_red[j,i]>0 or mask_blue[j,i]>0 or mask_yellow[j,i]>0):
                last = j    
   return last
f= open("new_to_curzon1.txt","a+")

imgNo=0

for img in glob.glob("E:/Images/*.png"):            
#    print("Reading image no ",imgNo)
    imgNo+=1
    inputImage = cv2.imread(str(img))
    f.write('\n')
    arr = str(img)[:-4].split('-')               
    f.write(arr[0][10:])
    f.write(",")
    if(arr[0]=='Fri' or arr[0]=='Sat'):
        f.write("Yes,")
    else:
        f.write("No,")
    #f.write(str(inputImage))
    if(arr[2]=="60"):
        f.write(str(int(arr[1])+1))
        f.write(".")
        f.write("00")
        f.write(',')
    else:    
        f.write(arr[1])
        f.write(".")
        f.write(arr[2])
        f.write(',')    
    #    cv2.imshow('final',inputImage)
    #    cv2.waitKey(0)
    #    cv2.destroyAllWindows()
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
    
    start = findInit(width,height)
    finish = findFinish(width,height)
    
    for j in range (start,height,40):
    #          print(j)  
          red=0
          blue=0
          yellow=0
          for k in range (j,j+40):
                for i in range (0,width):
                     if(k>finish):
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
                     if(mask_red[k,i]>0):
                         red = red+1
                    
                     if(mask_blue[k,i]>0):
                         blue = blue+1    
                     if(mask_yellow[k,i]>0):
                         yellow = yellow+1


#          print("Seg-->",seg,red,"  ",blue," ",yellow,"\n")  
          if(yellow>red and yellow>blue):
              f.write("Y,")
          elif(red>yellow and red>blue):
              f.write("R,")
          elif(blue>red and blue>yellow):
              f.write("B,")
          else:
              print("Image ",imgNo," Segment ",seg, "kisu korte pari nai")
          seg = seg+1
          