# -*- coding: utf-8 -*-
"""
Created on Sat Apr 20 20:14:37 2019

@author: User
"""
'''
import pyautogui as pa 
import time
import cv2 
import numpy as np
x=1

while(x<=5):
    pa.screenshot('scr'+str(x)+'.png')
    x+=1
    time.sleep(5)
'''


# import the necessary packages
import numpy as np
import cv2
import glob


for img in glob.glob("E:/Images/*.png"): 
    image = cv2.imread(str(img))
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    ############ For Yellow Color ############
    
    lower_yellow = np.array([10,160,240])
    upper_yellow = np.array([80,250,255])
    
    mask = cv2.inRange(hsv,lower_yellow,upper_yellow)
    res_yellow = cv2.bitwise_and(image,image, mask = mask)
    cv2.imwrite('messi_yellow.png',res_yellow)
    
    
    ######### For Blue Color##############
    
    lower_blue = np.array([100,90,150]) 
    upper_blue = np.array([145,255,255]) 
    mask = cv2.inRange(hsv, lower_blue, upper_blue) 
    res_blue = cv2.bitwise_and(image,image, mask=mask) 
    cv2.imwrite('messi_blue.png',res_blue)
    
    
    
    #########  For Red Color ###########
    lower_red = np.array([150,150,150]) 
    upper_red = np.array([255,255,255]) 
    
    mask = cv2.inRange(hsv, lower_red, upper_red) 
    
    res_red = cv2.bitwise_and(image,image, mask= mask)
    #cv2.imwrite('messi_red.png',res_red)
    #cv2.imshow('red',cv2.imread('messi_red.png'))
    #cv2.waitKey(0)
    cv2.destroyAllWindows()
    height, width, channels = image.shape
    
    res = res_red+res_blue+res_yellow
    for i in range(0,height):
        for j in range(0,width):
            a,b,c = res_red[i,j]
            if(a>0 or b>0 or c>0):
                 a,b,c = res_blue[i,j]
                 if(a>0 or b>0 or c>0):   
                     a,b,c = res_blue[i,j]
                     if(a>0 or b>0 or c>0):    
                         res[i,j] = 0,0,0
                         
    res[:,850:] = [0,0,0]
    res[:,0:640] = [0,0,0]
    res[350:,:700]  = [0,0,0]
    kernel = np.ones((20,1), np.uint8)  # note this is a horizontal kernel
    d_im = cv2.dilate(res, kernel, iterations=1)
    e_im = cv2.erode(d_im, kernel, iterations=1) 
    cv2.imwrite(str(img),res)
#    cv2.imshow(str(img),image)
    
    #cv2.imshow('res',cv2.imread('messi.png')) 
    ##cv2.imshow('im1',cv2.imread('messi_red_blue.png'))
    ##print(image2[100,100])
    #
    #
    ##
    #final = cv2.imread('messi_red_blue.png')
    #gray = cv2.cvtColor(final, cv2.COLOR_BGR2GRAY)
    #edged = cv2.Canny(gray, 50, 100)
    ##
    ##contours = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    ##contours = imutils.grab_contours(contours)
    #mask = np.ones(final.shape[:2], dtype="uint8") * 255
    #contours = cv2.findContours(gray.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[1]
    #contour=max(contours, key = cv2.contourArea)
    #cv2.drawContours(mask,[contour],-1,0,1)
    ## 
    #print("a")
    #print(cv2.contourArea(contour))
    #print("a")
    ### loop over the contours
    ##for c in contours:
    ##	# if the contour is bad, draw it on the mask
    ##	if(cv2.contourArea(c)<1000):
    ##		cv2.drawContours(mask, [c], -1, 0, -1)
    ###
    ##final = cv2.bitwise_and(final,final,mask = mask)
    ##rect = cv2.minAreaRect(contour)
    ##box = cv2.boxPoints(rect)
    ##box = np.int0(box)
    ##cv2.drawContours(image2,contour,0,(255,0,0),3)
    ##cv2.imshow('img',image2)
    ##
    #
    #cv2.imshow('final',final)
#    cv2.waitKey(0)
#    cv2.destroyAllWindows()
#    
