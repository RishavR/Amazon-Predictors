import pandas as pd 
import math
from textblob import TextBlob as tb

dataset=pd.read_csv("Feature_Set_Data_Filtered.csv")
reviewset=dataset["Reviews"].tolist()
tfidf_set= pd.DataFrame(columns=['Reviews','tfidf'])
bloblist=[]
for review in reviewset:
    bloblist.append(tb(review))

def tf(word, blob):
    return blob.words.count(word) / len(blob.words)

def n_containing(word, bloblist):
    return sum(1 for blob in bloblist if word in blob.words)

def idf(word, bloblist):
    return math.log(len(bloblist) / (1 + n_containing(word, bloblist)))

def tfidf(word, blob, bloblist):
    return tf(word, blob) * idf(word, bloblist)

for i, blob in enumerate(bloblist):
    print("Review No:",i)
    scores = {word: tfidf(word, blob, bloblist) for word in blob.words}
    sorted_words = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    tfidfC=0
    for word, score in sorted_words[:3]:
        tfidfC=tfidfC+round(score,5)
    tfidf_set.loc[i]=[blob,tfidfC]

final_feature_set = pd.merge(left=dataset,right=tfidf_set, left_on='Reviews', right_on='Reviews')

final_feature_set.to_csv("Feature_Set_Data_Filtered_TFIDF.csv",idex=False)