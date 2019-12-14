import cv2
import pandas as pd
import numpy as np # linear algebra
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
dataset  = pd.read_csv(r'data.csv',sep=',',error_bad_lines=False)
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

def definePixels(s,f,w):
   print(s,f) 
   fi= open("pixels3.txt","a+")
   for j in range (s,f):
        for i in range (0,w):
             if(mask_red[j,i]>0 or mask_blue[j,i]>0 or mask_yellow[j,i]>0):
                 fi.write(str(j))
                 fi.write(',')
                 fi.write(str(i))
                 fi.write('\n')
   fi.close()              
            
     
#Y = dataset.iloc[:,pos+1]
predictions = []

for pos in range(4,dataset.shape[1]):
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
    from sklearn.metrics import accuracy_score
    
    knn = KNeighborsClassifier(n_neighbors = 5, p = 2, metric='minkowski')
    #knn.fit(X_train_std, y_train)
    knn.fit(X, Y)
#    predicted = knn.predict([[3,0,2,12.25]]);
#    acc = accuracy_score(Y, predicted)
    predictions.append(knn.predict([[3,0,2,0.25]]))  
#    print('The accuracy of the Knn  classifier on training data is {:.2f}'.format(knn.score(X_train_std, y_train)))
#    print('The accuracy of the Knn classifier on test data is {:.2f}'.format(knn.score(X_test_std, y_test)))


inputImage = cv2.imread("F:\JAMILPC\Oct-Fri-0-15.png")
inputImage1 = inputImage
with open('pixels3.txt', 'r') as file:
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
#definePixels(start,finish,width)       


for line in range (0,len(data)):
    X = int(data[line].split(",")[0])
    Y = int(data[line].split(",")[1])
    seg = -1
    for f in range(0,10):
        if(X>=start+40*f and X<start+40*(f+1)):
            seg = f
    if(seg==-1):
        seg = 9
        
    if(predictions[seg]=='Y'):
        inputImage1[X,Y] = (0,255,255)
    if(predictions[seg]=='R'):
        inputImage1[X,Y] = (0,0,255)
    if(predictions[seg]=='B'):
        inputImage1[X,Y] = (255,0,0)
    if(len(data)%50==0):        
        print("Working on line ",line)    
'''
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
'''
#inputImage1 = inputImage
#for i in range(0,100):
#    for j in range(0,100):
#        inputImage1[i,j]=(0,0,0)
#      
cv2.imshow('final',inputImage1)
cv2.waitKey(0)
cv2.destroyAllWindows()        


#    predicted = knn.predict([[3,0,2,12.25]]);
#    acc = accuracy_score(Y, predicted)
  
        
#    print('The accuracy of the Knn  classifier on training data is {:.2f}'.format(knn.score(X_train_std, y_train)))
#    print('The accuracy of the Knn classifier on test data is {:.2f}'.format(knn.score(X_test_std, y_test)))

