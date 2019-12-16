import requests
from bs4 import BeautifulSoup as bs
import sys
import cv2
import pandas as pd
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier

import os 

date = sys.argv[1].strip()
import datetime
month, day, year = (int(x) for x in date.split('-'))    
ans = datetime.date(month, day, year)
val = ans.strftime("%A")

path = "\\resources\\app\\engine\\"
if(val=="Friday" or val=="Saturday"):
    val = 1
elif(val=="Sunday" or val=="Thursday"):
    val = 2
else:
    val = 3

holiday = sys.argv[4].strip()

if(holiday=="true"):
    holiday = 1

if(holiday=="false"):
    holiday = 0


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


time1 = sys.argv[2].strip()
time1 = time1.split(":")[0]+"."+time1.split(":")[1]
time1 = float(time1)
time1 = int(time1)+(int(str(time1).split(".")[1])*60)/100
# print(time1)

time2 = sys.argv[3].strip()
time2 = time2.split(":")[0]+"."+time2.split(":")[1]
time2 = float(time2)
time2 = int(time2)+(int(str(time2).split(".")[1])*60)/100
# print(time2)

# print(val,time1,time2)
swap = time1
if(time1>time2):
    time1 = time2
    time2 = swap
    swap = time2
# print(swap)    
cd = os.getcwd()
# print(date,time1,time2)
dataset  = pd.read_csv(cd+path+"data.csv")
pos = 3
X = dataset.iloc[:,:4]


predictions = []
time_results = []

# print(val,holiday)

while time1<time2:         
    tmp = []    
    for pos in range(4,dataset.shape[1]):
        Y = dataset.iloc[:,pos]

        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=0) 
        
        knn = KNeighborsClassifier(n_neighbors = 9, p = 2, metric='minkowski')
        knn.fit(X, Y)
        tmp.append(knn.predict([[3, holiday, val,time1]]))

    time1+=0.25
    if(int(str(time1).split(".")[1])>=60):
        time1 = int(time1)+1+0.00
    predictions.append(tmp)


for pos in range(4,dataset.shape[1]):
        Y = dataset.iloc[:,pos]

        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=0) 
        
        knn = KNeighborsClassifier(n_neighbors = 5, p = 2, metric='minkowski')
        knn.fit(X, Y)
        tmp.append(knn.predict([[3, holiday, val,time2]]))

summations = []
for i in range(0,len(predictions)):
    summations.append(0)

for i in range(0,len(predictions)):    
    for j in range(0,11):
        if(predictions[i][j-1]=='Y'):
            summations[i]+=4
        if(predictions[i][j-1]=='R'):
            summations[i]+=8
        if(predictions[i][j-1]=='B'):
            summations[i]+=2
        
# print(summations)
str1 = str(swap+(0.25*summations.index(min(summations))+1))
str1 = str1+" " 

inputImage = cv2.imread(cd+path+"1-Oct-Tue-0-15.png")
inputImage1 = inputImage

with open(cd+path+'pixels2.txt', 'r') as file:
    data = file.readlines()

seg = 0
height, width, channels = inputImage.shape
hsv = cv2.cvtColor(inputImage, cv2.COLOR_BGR2HSV)

lower_yellow = np.array([10,150,200])
upper_yellow = np.array([80,250,255])

lower_blue = np.array([100,90,150]) 
upper_blue = np.array([145,255,255]) 

lower_red = np.array([150,150,150]) 
upper_red = np.array([255,255,255]) 
mask_red =  cv2.inRange(hsv, lower_red, upper_red)
mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)   
    

start = findInit(width,height)
finish = findFinish(width,height)
minIndex = summations.index(min(summations))

for line in range (0,len(data)):
    X = int(data[line].split(",")[0])
    Y = int(data[line].split(",")[1])
    seg = -1
    for f in range(0,11):
        if(X>=start+40*f and X<start+40*(f+1)):
            seg = f
        if(seg==-1):
            seg = 10
            
    if(predictions[minIndex][seg-1]=='Y'):
        inputImage1[X,Y] = (0,255,255)
    if(predictions[minIndex][seg-1]=='R'):
        inputImage1[X,Y] = (0,0,255)
    if(predictions[minIndex][seg-1]=='B'):
        inputImage1[X,Y] = (255,0,0)


import base64
success, encoded_image = cv2.imencode('.png', inputImage1)
content = encoded_image.tobytes()
print(str1+base64.b64encode(content).decode('ascii'))
cv2.imshow('final',inputImage1)

sys.stdout.flush()






