import pyautogui as pa
import time
import cv2 
import numpy as np
x=1

while(x<=5):
    pa.screenshot('scr'+str(x)+'.png')
    x+=1
    time.sleep(5)