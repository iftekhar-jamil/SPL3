
data = []
with open(r'inter2.txt', 'r') as file:
    # read a list of lines into data
    data = file.readlines()

for i in range (0,len(data)):
    
     tmp = data[i][0:-1]
     arr = tmp.split(",")
     
     if("Jul" in tmp):
         arr[1] = "1"
     
     elif("Aug" in tmp):
         arr[1] = "2"
     
     elif("Sep" in tmp):
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
     if("Yes" in tmp):
         arr[2] = "1"
     
     else:
         arr[2] = "0"   
                 
     if(arr[3]=="Fri" or arr[3]=="Sat"):
         arr[3] = "0"
     
     if(arr[3]=="Sun" or arr[3]=="Thu"):
         arr[3] = "1"  
     else:
         arr[3] = "2"
     arr[3] = "2"    
#        
     tmp=""    
     for k in range(1,16):
         tmp+=arr[k]
         if(tmp==len(arr)-1):
             continue
         tmp+=","
     tmp = tmp[:-3]
     print(tmp)
#     tmp = tmp[:-1]
     data[i] = tmp+'\n'        

with open('data1.csv', 'w') as file:
    file.writelines( data )    