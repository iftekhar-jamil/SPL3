# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 00:17:47 2019

@author: User
"""
import cv2 

img = cv2.imread('scr3.png')



crop_img = img[290:580, 0:1350]
cv2.imwrite('cropped.png',crop_img)
cv2.imshow("cropped", crop_img)
cv2.waitKey(0)
cv2.destroyAllWindows()