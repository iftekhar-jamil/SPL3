
data = []
with open(r'intermediate22.txt', 'r') as file:
    # read a list of lines into data
    data = file.readlines()

for i in range (0,len(data)):
    
     tmp = data[i][0:-1]
     arr = tmp.split(",")
     if(arr[1]=="Jul"):
         arr[1] = "1"
         
     if(arr[1]=="Jul"):
         arr[1] = "1"
     if(arr[1]=="Aug"):
         arr[1] = "2"
     if(arr[1]=="Sep"):
         arr[1] = "3"
     else:
         arr[1] = "4"
#    month = tmp[3:6]
#    
#    if(month=="Jul"):
#        tmp = tmp.replace(month,"1")
#   
#    if(month=="Aug"):
#        tmp = tmp.replace(month,"2")
#     
#    if(month=="Sep"):
#        tmp = tmp.replace(month,"3")
#        
#    if(month=="Oct"):
#        tmp = tmp.replace(month,"4")
#         
#         
     holiday = arr[2]
     if(holiday=="No"):
        arr[2] = "0"
        
     else:
        arr[2] = "1"
     
     if(arr[3]=="Fri" or arr[3]=="Sat"):
         arr[3] = "0"
     
     if(arr[3]=="Sun" or arr[3]=="Thu"):
         arr[3] = "2"  
     else:
         arr[3] = "3"
#        
     tmp=""    
     for k in range(1,len(arr)):
         tmp+=arr[k]
         if(tmp==len(arr)-1):
             continue
         tmp+=","
     tmp = tmp[:-3]
     print(tmp)
#     tmp = tmp[:-1]
     data[i] = tmp+'\n'        

with open('data.csv', 'w') as file:
    file.writelines( data )    