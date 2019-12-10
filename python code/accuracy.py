import cv2
import pandas as pd
import numpy as np # linear algebra
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

dataset  = pd.read_csv(r'data.csv',sep=',',error_bad_lines=False)
pos = 3
X = dataset.iloc[:,:4]


dataset  = pd.read_csv(r'data.csv',sep=',',error_bad_lines=False)
pos = 3
X = dataset.iloc[:,:4]

with open(r'test.txt', 'r') as file:
    # read a list of lines into data
    data = file.readlines()

values = [[]]
for i in range (0,len(data)):
    tmp = data[i][0:-1]
    arr = tmp.split(',')
    values.append([])
    for k in range(4, len(arr)):
        values[i].append(arr[k])        
        
true = 0
false = 0        
for i in range(0,24):
    for pos in range(4,dataset.shape[1]):
        Y = dataset.iloc[:,pos]
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=0) 

        
        sc = StandardScaler()
        sc.fit(X_train)
        X_train_std = sc.transform(X_train)
        X_test_std = sc.transform(X_test)
        

        knn = KNeighborsClassifier(n_neighbors = 5, p = 2, metric='minkowski')
        knn.fit(X, Y)
#        print("Prediction for time ", float(i))
#        print(knn.predict([[3,0,2,float(i)]]))
        if(knn.predict([[3,0,2,float(i)]])==values[i][pos-4]):
            true+=1
        else:
            false+=1
            
accuracy = float(true)/float(false+true)            
print('True values ', true)
print('False values ', false)
print("Accuracy ", float(true)/float(false+true))
             
            