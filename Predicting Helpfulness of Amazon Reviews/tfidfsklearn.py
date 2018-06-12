import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
# list of text documents
dataset=pd.read_csv("Feature_Set_Data_Filtered.csv")
text=dataset["Reviews"].tolist()
tfidf_set= pd.DataFrame(columns=['Reviews','Sig(tfidf)'])
# create the transform
vectorizer = TfidfVectorizer()
# tokenize and build vocab
vectorizer.fit(text)
# summarize
print(vectorizer.vocabulary_)
print(vectorizer.idf_)
# encode document
i=0
for review in text:
    vector = vectorizer.transform([review])
# summarize encoded vector
    print("Review Count:",i)
    vectorlist=vector.toarray() 
    tfidf_set.loc[i]=[review,vectorlist.sum(axis=1)]
    i=i+1
merged_inner = pd.merge(left=dataset,right=tfidf_set, left_on='Reviews', right_on='Reviews')
merged_inner.to_csv("Feature_Set_Data_FilteredTFIDF.csv",index=False)    