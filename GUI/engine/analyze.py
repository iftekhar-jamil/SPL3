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

time1 = 00.00
date = 3
cd = os.getcwd()

date = sys.argv[1]
import datetime
month, day, year = (int(x) for x in date.split('-'))    
ans = datetime.date(month, day, year)
val = ans.strftime("%A")

if(val=="Friday" or val=="Saturday"):
    val = 1

elif(val=="Sunday" or val=="Thursday"):
    val = 2

else:
    val = 3


holiday = sys.argv[2].strip()

if(holiday=="true"):
    holiday = 1

if(holiday=="false"):
    holiday = 0

date = 3
print(date, holiday, val, time1)

dataset  = pd.read_csv(cd+"/engine/data.csv")
pos = 3
X = dataset.iloc[:,:4]


# predictions = [[]]
predictions = []
time_results = []
for pos in range(4,dataset.shape[1]):
    Y = dataset.iloc[:,pos]
#    pos = pos+1
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=0) 
    
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import cross_val_score
    
   
    from sklearn.neighbors import KNeighborsClassifier
    
    knn = KNeighborsClassifier(n_neighbors = 5, p = 2, metric='minkowski')
    #knn.fit(X_train_std, y_train)
    knn.fit(X, Y)
    tmp = []
    for i in range(0,13):         
        tmp.append(knn.predict([[int(date), int(holiday), int(val), time1]]))
        time1+=2.00
    # print(tmp[0]) 
    # print(len(tmp))   
    predictions.append(tmp)
summations = []
for i in range(0,13):
    summations.append(0)

print(predictions)
for i in range(0,13):    
    for j in range(0,11):
        if(predictions[j-1][i]=='Y'):
            summations[i]+=20
        if(predictions[j-1][i]=='R'):
            summations[i]+=30
        if(predictions[j-1][i]=='B'):
            summations[i]+=10
        

print(summations)
sys.stdout.flush()

