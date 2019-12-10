import numpy as np
import cv2
import glob

imgNo = 0
for img in glob.glob("F:/Data/Friday+Saturday/*.png"): 
    image = cv2.imread(str(img))
    imgNo += 1
    print("Reading image no ",imgNo)
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
