import pandas as pd
import numpy as np 
from sklearn.metrics import accuracy_score
dataset= pd.read_csv("Feature_Set_Data_FilteredTFIDF.csv")
dataset.drop(['Product Name','Reviews'],axis=1,inplace=True)
Score_storage=[]
reviewSize=5000
trainSize=0.8
kernel_type='poly'
#Calculating the Average number of 'Helpful' review votes
condition= dataset["Review Votes"].sum()/dataset["Review Votes"].count()
#Checking each Review Vote to see if it satisfies the condition. If it does assign 1 to 'Useful' column else assign 0
dataset['Useful'] = (dataset['Review Votes'] > condition).astype(int)
useful_Reviews= dataset[dataset['Useful'] == 0]
unuseful_Reviews=dataset[dataset['Useful'] == 1]

randomSelector=np.random.choice(np.arange(len(useful_Reviews)), size=int((reviewSize)/2))
x1 = dataset.iloc[randomSelector,0:10].values
y1 = dataset.iloc[randomSelector,10].values

randomSelector2=np.random.choice(np.arange(len(unuseful_Reviews)), size=int((reviewSize)/2))
x2 = dataset.iloc[randomSelector2,0:10].values
y2 = dataset.iloc[randomSelector2,10].values

x=np.concatenate((x1,x2),axis=0)
y=np.concatenate((y1,y2),axis=0)

from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = trainSize, random_state = 0) 
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)
#Finding Absolute Value- Only required for MultinomialGB
x_train = np.absolute(x_train)
x_test = np.absolute(x_test)
y_train = np.absolute(y_train)
y_test = np.absolute(y_test)
#######Linear SVM Classification:

# Fitting SVM to the Training set
#from sklearn.svm import SVC
#classifier = SVC(kernel = 'linear', random_state = 0)
####RBF SVM Classification:
#classifier = SVC(kernel = 'rbf', random_state = 0)
###### POLYNOMIAL########
#classifier = SVC(kernel = 'poly', random_state = 0, degree = 3)
###### 4 Kernel Classifier ############################
#classifier = SVC(kernel = kernel_type, random_state = 0)
###### Gaussian Naive Bayes Classifier COMMENT OUT IF NOT REQUIRED ###############
#from sklearn.naive_bayes import GaussianNB
#classifier=GaussianNB()
###### Multinomial Naive Bayes Classifier COMMENT OUT IF NOT REQUIRED ###############
from sklearn.naive_bayes import MultinomialNB
classifier=MultinomialNB()


classifier.fit(x_train, y_train)
predicted_test=classifier.predict(x_test)
predicted_train=classifier.predict(x_train)
#get the accuracy score
test_accuracy=accuracy_score(y_test,predicted_test)
train_accuracy=accuracy_score(y_train,predicted_train)
print(test_accuracy,train_accuracy)
dataStore=pd.read_csv("testResult.csv")

tempdataFrame=pd.DataFrame([[kernel_type,trainSize,train_accuracy,test_accuracy]],columns=['Kernel Type','Training Size','Train Accuracy','Test Accuracy'])
dataStore=dataStore.append(tempdataFrame,ignore_index=True)
dataStore.to_csv("testResult.csv",index=False)

