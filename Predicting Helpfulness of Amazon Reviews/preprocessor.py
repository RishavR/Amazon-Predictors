import pandas as pd 

dataset=pd.read_csv("Amazon_Unlocked_Mobile.csv")
#print (dataset)
#X= dataset.iloc[:,[3,4,5]].values
#print(X)
dataset.drop(['Brand Name','Price'],axis=1,inplace=True)
#Replaces in place without reassignment 
#print(dataset)
grouped=dataset.groupby(['Product Name']).count()
grouped= grouped[grouped["Review Votes"] > 10]
#print(grouped)
grouped2=dataset.groupby(['Product Name']).sum()
#print(grouped2)
finaldat=grouped[['Rating','Review Votes']].copy()
finaldat.insert(column="Avg Ratings",loc=2,value=grouped2['Rating']/grouped['Rating'])
finaldat['Product Name']=finaldat.index
finaldat.drop(['Rating','Review Votes'],axis=1,inplace=True)
merged_inner = pd.merge(left=dataset,right=finaldat, left_on='Product Name', right_on='Product Name')
#merged_inner= merged_inner[merged_inner["Review Votes"] > 2]
merged_inner.dropna(inplace=True)
merged_inner.drop_duplicates(subset="Reviews", inplace=True)
merged_inner.to_csv("Modified_data_NonFiltered.csv",index=False)

