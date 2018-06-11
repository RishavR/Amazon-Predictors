import pandas as pd 
import nltk as nlp 
import numpy as np 

dataset=pd.read_csv("Modified_data_NonFiltered.csv")
#Convert to a list of str reviews which can be manipulated by nltk
review=dataset["Reviews"].tolist()
#Creating a new DataFrame which will be used to store features extracted from each sentence 
feature_set= pd.DataFrame(columns=['Length','sentenceCount','wordCount','characterCount','symbolicQuotient','FRES','ARI','Deviation'])
# A very crude code to extract all features apart from tfidf (we use sklearn for that)
#Change ASAP 

for sentence in review:
    sentenceCount=len(nlp.sent_tokenize(sentence))
    wordCount=len(nlp.word_tokenize(sentence))
    characterCount=len(sentence)
    syllableCount=0
    from nltk.corpus import cmudict
    d = cmudict.dict()
    for word in word_bank:
        syllableCount=syllableCount+nsyl(word)
    def nsyl(word):
        return [len(list(y for y in x if y[-1].isdigit())) for x in d[word.lower()]
    questionMarkCount= len(my_text) - len(my_text.rstrip('?'))
    exclamationCount=
    symolicQuotient= exclamationCount-questionMarkCount
    #Calculatrs the Flesch Reading Ease Score 
    FRES= 206.835 -1.015*(float(wordCount)/sentenceCount) - 84.6*(float(syllableCount)/wordCount)
    #Calculates the Automated Readibility Index
    ARI= 4.71(float(characterCount)/wordCount) + 0.5*(float(wordCount)/sentenceCount) - 21.43
    