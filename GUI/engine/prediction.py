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
import os 


time = sys.argv[1].strip()
time = time.split(":")[0]+"."+time.split(":")[1]
time = float(time)

date = sys.argv[2].strip()
# print(date)
import datetime
# dt = '15/12/2019'
month, day, year = (int(x) for x in date.split('-'))    
ans = datetime.date(month, day, year)
val = ans.strftime("%A")

# date = date.split("-")[1]

# val="a"
if(val=="Friday" or val=="Saturday"):
    val = 1

elif(val=="Sunday" or val=="Thursday"):
    val = 2

else:
    val = 3

holiday = sys.argv[3].strip()

if(holiday=="true"):
    holiday = 1

if(holiday=="false"):
    holiday = 0

date = 3

# print(date, holiday, val, time)
cd = os.getcwd()
#print(os.listdir())
dataset  = pd.read_csv(cd+"/engine/data.csv")
pos = 3
X = dataset.iloc[:,:4]


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
#Y = dataset.iloc[:,pos+1]
predictions = []

for pos in range(4,dataset.shape[1]):
    Y = dataset.iloc[:,pos]

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=0) 
   
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import cross_val_score
    
    # sc = StandardScaler()
    # sc.fit(X_train)
    # X_train_std = sc.transform(X_train)
    # X_test_std = sc.transform(X_test)
    
    #Applying Knn
    from sklearn.neighbors import KNeighborsClassifier
    
    knn = KNeighborsClassifier(n_neighbors = 5, p = 2, metric='minkowski')
    #knn.fit(X_train_std, y_train)
    knn.fit(X, Y)
    predictions.append(knn.predict([[int(date), int(holiday), int(val), time]]))  


inputImage = cv2.imread(cd+"/engine/1-Oct-Tue-0-15.png")
inputImage1 = inputImage

with open(cd+'/engine/pixels2.txt', 'r') as file:
    # read a list of lines into data
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

# print(len(predictions))
for line in range (0,len(data)):
    X = int(data[line].split(",")[0])
    Y = int(data[line].split(",")[1])
    seg = -1
    for f in range(0,11):
        if(X>=start+40*f and X<start+40*(f+1)):
            seg = f
        if(seg==-1):
            seg = 10
            
    if(predictions[seg-1]=='Y'):
        inputImage1[X,Y] = (0,255,255)
    if(predictions[seg-1]=='R'):
        inputImage1[X,Y] = (0,0,255)
    if(predictions[seg-1]=='B'):
        inputImage1[X,Y] = (255,0,0)
    # if(len(data)%50==0):        
    #     print("Working on line ",line)    

import base64

success, encoded_image = cv2.imencode('.png', inputImage1)
content = encoded_image.tobytes()
decoded_image = base64.b64encode(content).decode('ascii')
print(decoded_image)
cv2.imshow('final',inputImage1)


#print(city,city1)
sys.stdout.flush()
