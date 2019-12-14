import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
#path = [cv2.imread(file) for file in glob.glob("F:/Projects/Python Project/Images/*.png")]

lower_yellow = np.array([10,150,200])
upper_yellow = np.array([80,250,255])

lower_blue = np.array([100,90,150]) 
upper_blue = np.array([145,255,255]) 

lower_red = np.array([150,150,150]) 
upper_red = np.array([255,255,255]) 
images = []

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
f= open("inter4.txt","a+")
#pixels = open("pixels3.txt","a+")
Colored_Pixels = []

imgNo=0

for img in glob.glob("F:/Data/Others/*.png"):            
    print("Reading image no ",imgNo)
    imgNo+=1
#    if(imgNo==2):
#        break
    inputImage = cv2.imread(str(img))
#    if(str(img).find("Sep")==-1):
#        continue
    
    
    arr = str(img)[:-4].split('-')   
    arr1 =  arr[0][8:].split(' ')
#    if(arr[1]!='Fri'): 
#        continue    
    f.write('\n')               
    f.write(arr1[0])
    f.write(",")
    f.write(arr1[1])
    f.write(",")
    if(arr[1]=='Fri' or arr[1]=='Sat'):
        f.write("Yes,")
    else:
        f.write("No,")
    f.write(arr[1])    
    f.write(',')
#    f.write(arr[2])    
#    f.write(',')
#    f.write(arr[3])    
#    f.write(',')
    
#    f.write(arr[4])
    #f.write(str(inputImage))
    if(arr[3]=="60"):
        f.write(str(int(arr[2])+1))
        f.write(".")
        f.write("00")
        f.write(',')
    else:    
        f.write(arr[2])
        f.write(".")
        f.write(arr[3])
        f.write(',')    
    #    cv2.imshow('final',inputImage)
    #    cv2.waitKey(0)
    #    cv2.destroyAllWindows()
    hsv = cv2.cvtColor(inputImage, cv2.COLOR_BGR2HSV)
    #plt.imshow(inputImage)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    mask_red =  cv2.inRange(hsv, lower_red, upper_red)
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)   
    
    inputImageGray = cv2.cvtColor(inputImage, cv2.COLOR_BGR2GRAY)
    seg = 1
    height, width, channels = inputImage.shape
    
    start = findInit(width,height)
    finish = findFinish(width,height)
    
    for j in range (start,height,40):
    #          print(j)  
          red=0
          blue=0
          yellow=0
          for k in range (j,j+40):
                for i in range (0,width):
                     if(k>finish):
                        break
        #             a,b,c =  inputImage[i,k];
        #            if(a==0 and b==0 and c==0):
        #                continue;
        #            if(isInRange(a,b,c)==0):
        #                y = y+1
        #            elif(isInRange(a,b,c)==1):
        #                r = r+1
        #            else:
        #                b = b+1
                        
                     if(mask_red[k,i]>0):
                         red = red+1
                         if(imgNo==1):
                             Colored_Pixels.append([k,i])
                         
                     if(mask_blue[k,i]>0):
                         blue = blue+1                         
                         if(imgNo==1):
                             Colored_Pixels.append([k,i])
                         
                     if(mask_yellow[k,i]>0):
                         yellow = yellow+1
                         if(imgNo==1):
                             Colored_Pixels.append([k,i])
                             
#          print("Seg-->",seg,red,"  ",blue," ",yellow,"\n")  
          if(yellow>red and yellow>blue):
              f.write("Y,")
          elif(red>yellow and red>blue):
              f.write("R,")
          elif(blue>red and blue>yellow):
              f.write("B,")
#          else:
#              print("Image ",imgNo," Segment ",seg, "kisu korte pari nai")
          seg = seg+1

f.close()
#pixels.close()


#pixels = open("pixels2.txt","a+")
#
#for i in range(0,len(Colored_Pixels)):
#    pixels.write(str(Colored_Pixels[i][0])+","+str(Colored_Pixels[i][1])+"\n") 
#pixels.close()        