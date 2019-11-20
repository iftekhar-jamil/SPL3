import cv2
import pandas as pd
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import math
dataset  = pd.read_csv('data.csv')
pos = 3
X = dataset.iloc[:,2:3]

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
       
    
    #Create tree object
    decision_tree = tree.DecisionTreeClassifier(criterion='gini')
    
    #Train DT based on scaled training set
    decision_tree.fit(X, Y)
    predictions.append(decision_tree.predict([[11.15]]))    
    
    #Print performance
#    print('The accuracy of the Decision Tree classifier on training data is {:.2f}'.format(decision_tree.score(X_train_std, y_train)))
#    print('The accuracy of the Decision Tree classifier on test data is {:.2f}'.format(decision_tree.score(X_test_std, y_test)))
#

inputImage = cv2.imread("E:\ImagesNew\Thu-14-30.png")
inputImage1 = inputImage
with open('pixels2.txt', 'r') as file:
    # read a list of lines into data
    data = file.readlines()
seg = 0
height, width, channels = inputImage.shape

start = findInit(width,height)
finish = findFinish(width,height)


inputImage = cv2.imread("E:\ImagesNew\Thu-14-30.png")
inputImage1 = inputImage
with open('pixels2.txt', 'r') as file:
    # read a list of lines into data
    data = file.readlines()
seg = 0
height, width, channels = inputImage.shape

start = findInit(width,height)
finish = findFinish(width,height)


for line in range (0,len(data)):
    X_value = int(data[line].split(",")[0]
    Y_value = int(data[line].split(",")[1]
    seg = math.floor((X_value-start))
    if(predictions[seg]=='Y'):
        inputImage1[k,i] = (0,255,255)
    if(predictions[seg]=='R'):
        inputImage1[k,i] = (0,0,255)
    if(predictions[seg]=='B'):
        inputImage1[k,i] = (255,0,0)

for j in range (start,height,40):
#          print(j)  
      red=0
      blue=0
      yellow=0
      for k in range (j,j+40):
            for i in range (600,width):
                 if(k>finish):
                    break
                 for line in range (0,len(data)):
                     if(k==int(data[line].split(",")[0]) and i==int(data[line].split(",")[1]) and seg<11):
                         if(predictions[seg]=='Y'):
                             inputImage1[k,i] = (0,255,255)
                         if(predictions[seg]=='R'):
                             inputImage1[k,i] = (0,0,255)
                         if(predictions[seg]=='B'):
                             inputImage1[k,i] = (255,0,0)
                                             
      seg = seg+1
      print("Working on Segment ",seg)
#
#inputImage1 = inputImage
#for i in range(0,100):
#    for j in range(0,100):
#        inputImage1[i,j]=(0,0,0)
#      
cv2.imshow('final',inputImage1)
cv2.waitKey(0)
cv2.destroyAllWindows()        
    