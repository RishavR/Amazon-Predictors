import pandas as pd 
import re
from textblob import TextBlob as tb
from textstat.textstat import textstat

dataset=pd.read_csv("Modified_data.csv")
#Convert to a list of str reviews which can be manipulated by nltk
review=dataset["Reviews"].tolist()
#Creating a new DataFrame which will be used to store features extracted from each sentence 
feature_set= pd.DataFrame(columns=['Reviews','Length','sentenceCount','wordCount','characterCount','symbolicQuotient','FRES','ARI'])
#tfidf_set= pd.DataFrame(columns=['Reviews','tfidf'])
# A very crude code to extract all features apart from tfidf (we use sklearn for that)
#Change ASAP 
#review=review[:3]
counter=0
bloblist=[]
for sentence in review:
    blob=tb(sentence)
    #bloblist.append(blob)
    duplicateSentence=sentence
    print("Sentence No:",counter)
    sentenceCount=len(blob.sentences)
    if sentenceCount == 0:
        continue
    questionMarkCount = len(sentence) - len(sentence.rstrip('?'))
    exclamationCount= len(sentence) - len(sentence.rstrip('!'))
    symbolicQuotient= exclamationCount-questionMarkCount
    sentence= re.sub(r'[^\w]', ' ', sentence)
    word_bank=blob.words
    wordCount=len(word_bank)
    if wordCount == 0:
            continue
    characterCount=len(sentence)
    #Arbitary check to see if everything is okay
    print("\nSentence,Word,Character",sentenceCount,wordCount,characterCount)
    syllableCount=textstat.syllable_count(sentence)
        #syllableCount=syllableCount+len(nsyl(word)
    #Calculatrs the Flesch Reading Ease Score 
    FRES= textstat.flesch_reading_ease(sentence)
    #Calculates the Automated Readibility Index
    ARI= textstat.automated_readability_index(sentence)
    feature_set.loc[counter]= [duplicateSentence,characterCount,sentenceCount,wordCount,characterCount,symbolicQuotient,FRES,ARI]
    counter+=1
#Merging both the dataset and dataframe to get final feature frame. 
merged_inner = pd.merge(left=dataset,right=feature_set, left_on='Reviews', right_on='Reviews')
#final_feature_set = pd.merge(left=merged_inner,right=tfidf_set, left_on='Reviews', right_on='Reviews')
#Inserting column for deviation in final feature set
merged_inner.insert(column="Deviation",loc=2,value=abs(merged_inner['Rating']-merged_inner['Avg Ratings']))
merged_inner.drop(['Rating','Avg Ratings'],axis=1,inplace=True)
merged_inner.to_csv("Feature_Set_Data_Filtered.csv",index=False)


            
