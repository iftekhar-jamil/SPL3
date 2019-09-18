# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 18:55:34 2019

@author: User
"""
data = []
with open('data.csv', 'r') as file:
    # read a list of lines into data
    data = file.readlines()

for i in range (0,len(data)):
    tmp = data[i][0:-2]
    day = tmp[0:3]
    if(day=="Thu" or day=="Sun"):
        tmp = tmp.replace(day,"2")
    elif(day=="Fri" or day=="Sat"):
        tmp = tmp.replace(day,"1")
    else:
        tmp = tmp.replace(day,"0")
     
    holiday = tmp[2:4]
    if(holiday=="No"):
        tmp = tmp.replace(holiday,"1")
        
    else:
        tmp = tmp.replace(holiday,"0")
        
        

    print(tmp)
    data[i] = tmp+'\n'        

with open('data.csv', 'w') as file:
    file.writelines( data )    