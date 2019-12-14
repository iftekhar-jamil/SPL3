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

cd = os.getcwd()
#print(os.listdir())
dataset  = pd.read_csv(cd+"/engine/data.csv")
pos = 3
X = dataset.iloc[:,:3]


# predictions = [[]]
predictions = []
time_results = []
for pos in range(3,14):
    Y = dataset.iloc[:,pos]
#    pos = pos+1
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=0) 
    #test_size: if integer, number of examples into test dataset; if between 0.0 and 1.0, means proportion
#    print('There are {} samples in the training set and {} samples in the test set'.format(X_train.shape[0], X_test.shape[0]))
    
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import cross_val_score
    
    sc = StandardScaler()
    sc.fit(X_train)
    X_train_std = sc.transform(X_train)
    X_test_std = sc.transform(X_test)
    
    #Applying Knn
    from sklearn.neighbors import KNeighborsClassifier
    
    knn = KNeighborsClassifier(n_neighbors = 3, p = 2, metric='minkowski')
    #knn.fit(X_train_std, y_train)
    knn.fit(X, Y)
    tmp = []
    for i in range(0,13):         
        tmp.append(knn.predict([[0,1,time1]]))
        time1+=2.00
    # print(tmp[0]) 
    # print(len(tmp))   
    predictions.append(tmp)
summations = []
for i in range(0,13):
    summations.append(0)

# print(predictions[0])
for i in range(0,13):    
    for j in range(0,11):
        if(predictions[j][i]=='Y'):
            summations[i]+=2
        if(predictions[j][i]=='R'):
            summations[i]+=3
        if(predictions[j][i]=='B'):
            summations[i]+=1
        

print(summations)
# sys.stdout.flush()

