import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np 

dataset=pd.read_csv("testResult.csv")
dataset2=dataset.groupby(["Kernel Type","Training Size"]).sum()
print(dataset2)
dataset2["Train Accuracy"]=dataset2["Train Accuracy"]/3
dataset2["Test Accuracy"]=dataset2["Test Accuracy"]/3
dataset2=dataset2.reset_index()
#print(dataset2)
#dataset3=dataset2.drop(['Test Accuracy'],axis=1)
#dataset4=dataset2.drop(['Train Accuracy'],axis=1)
#################TRAIN ACCURACY PLOT #################
plt.figure(figsize=(16,12))
for i in range(0,42,7):
    x = dataset2.iloc[i:i+7,1].values
    y=dataset2.iloc[i:i+7,2].values
    print(str(dataset2.iloc[i,0]))
    plt.plot(x,y,label=str(dataset2.iloc[i,0]))
  
plt.legend(loc='best')   
plt.xlabel("Training Size")
plt.ylabel("Train Accuracy") 
#plt.show()
plt.savefig("TrainVsSize.png")
#################TEST ACCURACY ACCURACY PLOT #################
plt.figure(figsize=(16,12))
for i in range(0,42,7):
    x = dataset2.iloc[i:i+7,1].values
    y=dataset2.iloc[i:i+7,3].values
    print(str(dataset2.iloc[i,0]))
    plt.plot(x,y,label=str(dataset2.iloc[i,0]))
  
plt.legend(loc='best')   
plt.xlabel("Training Size")
plt.ylabel("Test Accuracy") 
#plt.show()
plt.savefig("TestVsSize.png")