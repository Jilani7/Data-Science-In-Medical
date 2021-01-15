#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Importing Libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# # Diabetes Dataset..

# ## Data Importing, Normalization and Data Discretization

# In[ ]:


# Reading data in csv

diabetes_df = pd.read_csv("Datasets/diabetes-data.txt",delimiter=',')


# In[ ]:


diabetes_df.head()


# ##  Diabetes Dataset Attribute: (all numeric-valued)
#    1. Number of times pregnant
#    2. Plasma glucose concentration a 2 hours in an oral glucose tolerance test
#    3. Diastolic blood pressure (mm Hg)
#    4. Triceps skin fold thickness (mm)
#    5. 2-Hour serum insulin (mu U/ml)
#    6. Body mass index (weight in kg/(height in m)^2)
#    7. Diabetes pedigree function
#    8. Age (years)
#    9. Class variable (0 or 1)

# In[ ]:


# Giving column Names as discussed in dataset description file

diabetes_df.columns = [
                        'PregnantTimes',
                        'PlasmaGlucose',
                        'DiastolicBlood',
                        'TricepsSkinFold',
                        'SerumInsulin',
                        'BMI',
                        'DiabetesPedigree',
                        'Age',
                        'Class'
                    ]


# In[ ]:


# Viwing first 10 rows

diabetes_df.head()


# In[ ]:


# Finding the shape of matrix 

diabetes_df.shape


# In[ ]:


# Repacing the '?' with Nan Values 

diabetes_df.replace("?", np.nan, inplace = True)


# In[ ]:


# Checking the null values of each column

diabetes_df.isnull().mean()


# In[ ]:


# Total Null values in each column

diabetes_df.isnull().sum()


# In[ ]:


# Types of each column of dataset

diabetes_df.info()


# In[ ]:


# There are many object(str) types converting them into float

diabetes_df = diabetes_df.astype(float)


# In[ ]:


# Filling the Nan values with Mean becuase droping columns will reduce the size of dataset and will cause underfitting

diabetes_df = diabetes_df.fillna(diabetes_df.mean())


# In[ ]:


# Varifying if all the Values Have been filled

diabetes_df.isnull().sum()


# In[ ]:


# Checking again the types to varify whether type is changed or not

diabetes_df.info()


# ## Feature Subset selection

# In[ ]:


# Converting output label back to int for only (0 or 1 values)

diabetes_df['Class'] = diabetes_df['Class'].astype(int)


# In[ ]:


# Summary statistics to know about Data minimum, maximum and Standard Deviation before selecting any feature

diabetes_df.describe()


# ### Standard Deviation of some of values is very high and showing bad behavior about data

# In[ ]:


# Checking how many columns have 0 values in dataset

diabetes_df.isin({0.0}).sum()


# In[ ]:


# Replacing all columns wiht mean expect pregnantTimes becuase there are some cases when patient has diabeties and hasn't been prgnent yet

diabetes_df['PlasmaGlucose'] = diabetes_df['PlasmaGlucose'].replace(0.0,diabetes_df['PlasmaGlucose'].mean())
diabetes_df['DiastolicBlood']=diabetes_df['DiastolicBlood'].replace(0.0,diabetes_df['DiastolicBlood'].mean())
diabetes_df['TricepsSkinFold']=diabetes_df['TricepsSkinFold'].replace(0.0,diabetes_df['TricepsSkinFold'].mean())
diabetes_df['SerumInsulin']=diabetes_df['SerumInsulin'].replace(0.0,diabetes_df['SerumInsulin'].mean())
diabetes_df['BMI']=diabetes_df['BMI'].replace(0.0,diabetes_df['BMI'].mean())


# In[ ]:


# Checking the 0's again in each column

diabetes_df.isin({0.0}).sum()


# In[ ]:


# Selecting all the features for our X values except label column and first we will predicit on it

X = diabetes_df.iloc[:, [0,1,2,3,4,5,6,7]]
X.head()


# In[ ]:


# Temperory Variable feature

Feature = X
Feature.head()


# In[ ]:


# Our Target output y

y = diabetes_df['Class'].values
y[0:5]


# In[ ]:


# Checking again the summary statistics. Our data is much better now

X.describe()


# In[ ]:


# Normalize the Dataset into a fix range of values 

from sklearn import preprocessing

X = preprocessing.StandardScaler().fit(X).transform(X)
X[0:5]


# ## Applying Classification Models

# In[ ]:


# Splitting the dataset using Scikit learning in 75 and 25 ratio for train and test respectivly


from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size = 0.25, random_state=0)
print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)


# In[ ]:


# Pridicting the dataset on different values of K from 1 to 20

k = 1
for i in range (20):
    knn = KNeighborsClassifier(n_neighbors = k).fit(X_train,Y_train)
    yhat = knn.predict(X_test)
    print ("KNN With k = " + str(k))
    print("Accuracy on train data with k = " + str (k) + " : ", metrics.accuracy_score(Y_train,knn.predict(X_train)) * 100)
    print("Accuracy on test data with  k = " + str(k) + " : ", metrics.accuracy_score(Y_test, yhat) * 100)
    print ()
    k+=1
    


# ## Best KNN Accuracy is on K = 13 with K Accuracy of Almost 80% 

# # Naive Bayes Algorithm

# In[ ]:


# Applying Bernoulli Naive Bayes

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0) 
from sklearn.naive_bayes import BernoulliNB 
gnb = BernoulliNB() 
gnb.fit(X_train, y_train) 
y_pred = gnb.predict(X_test) 
  

from sklearn import metrics
print("Bernoulli Naive Bayes model accuracy on test set     :", metrics.accuracy_score(y_test, y_pred)*100)


# In[ ]:


# Applying Gaussian Naive Bayes


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0) 
from sklearn.naive_bayes import GaussianNB 
gnb = GaussianNB() 
gnb.fit(X_train, y_train) 
y_pred = gnb.predict(X_test)   

from sklearn import metrics
print("Gaussian Naive Bayes model accuracy on test set     :", metrics.accuracy_score(y_test, y_pred)*100)


# ## Gaussian Naive Bayes is giving high accuracy among bernoulli and gaussian of 78%

# In[ ]:


# Plotting Correlation Matrix to see if some features are dominating on other features and eradicating uselesss features

plt.subplots(figsize=(10, 10))
plt.title('Correlation Matrix')
sns.heatmap(Feature.astype(float).corr(), square=True, annot=True)


# ## There are no conclusive features which are directly correlated so we can continue with all the features for X
# 

# In[ ]:





# # Hepatitis Dataset 

# In[ ]:


# Reading Hepatitis Dataset

hepatitis_df = pd.read_csv("Datasets/hepatitis-data.txt",delimiter=',')


# In[ ]:


# Visualizing first 5 rows

hepatitis_df.head()


# ## Attribute information from hepatitus-description.txt:
#      1. Class: DIE, LIVE
#      2. AGE: 10, 20, 30, 40, 50, 60, 70, 80
#      3. SEX: male, female
#      4. STEROID: no, yes
#      5. ANTIVIRALS: no, yes
#      6. FATIGUE: no, yes
#      7. MALAISE: no, yes
#      8. ANOREXIA: no, yes
#      9. LIVER BIG: no, yes
#     10. LIVER FIRM: no, yes
#     11. SPLEEN PALPABLE: no, yes
#     12. SPIDERS: no, yes
#     13. ASCITES: no, yes
#     14. VARICES: no, yes
#     15. BILIRUBIN: 0.39, 0.80, 1.20, 2.00, 3.00, 4.00
#         -- see the note below
#     16. ALK PHOSPHATE: 33, 80, 120, 160, 200, 250
#     17. SGOT: 13, 100, 200, 300, 400, 500, 
#     18. ALBUMIN: 2.1, 3.0, 3.8, 4.5, 5.0, 6.0
#     19. PROTIME: 10, 20, 30, 40, 50, 60, 70, 80, 90
#     20. HISTOLOGY: no, yes

# In[ ]:


# Giving Column Names Present in Dataset

hepatitis_df.columns = [
                        'Class',
                        'AGE',
                        'SEX',
                        'STEROID',
                        'ANTIVIRALS',
                        'FATIGUE',
                        'MALAISE',
                        'ANOREXIA',
                        'LIVER BIG',
                        'LIVER FIRM',
                        'SPLEEN PALPABLE',
                        'SPIDERS','ASCITES',
                        'VARICES','BILIRUBIN',
                        'ALK PHOSPHATE','SGOT',
                        'ALBUMIN','PROTIME',
                        'HISTOLOGY'
                    ]


# In[ ]:


# Visualising dataset with column names
hepatitis_df.head()


# In[ ]:


# Last 5 rows of dataset

hepatitis_df.tail()


# In[ ]:


# Checking the shape of dataset

hepatitis_df.shape


# In[ ]:


# Replcaing the '?' with nan values

hepatitis_df.replace("?", np.nan, inplace = True)


# In[ ]:


# Checking the output label count

hepatitis_df['Class'].value_counts()


# In[ ]:


# Mean of null rows of dataset

hepatitis_df.isnull().mean()


# In[ ]:


# Summary Statistics of dataset

hepatitis_df.describe()


# In[ ]:


# Checking the data types of dataframe column because less column shows in summary statistics

hepatitis_df.info()


# ## Attribute information from hepatitus-description.txt:
#      1. Class: DIE, LIVE
#      2. AGE: 10, 20, 30, 40, 50, 60, 70, 80
#      3. SEX: male, female
#      4. STEROID: no, yes
#      5. ANTIVIRALS: no, yes
#      6. FATIGUE: no, yes
#      7. MALAISE: no, yes
#      8. ANOREXIA: no, yes
#      9. LIVER BIG: no, yes
#     10. LIVER FIRM: no, yes
#     11. SPLEEN PALPABLE: no, yes
#     12. SPIDERS: no, yes
#     13. ASCITES: no, yes
#     14. VARICES: no, yes
#     15. BILIRUBIN: 0.39, 0.80, 1.20, 2.00, 3.00, 4.00
#         -- see the note below
#     16. ALK PHOSPHATE: 33, 80, 120, 160, 200, 250
#     17. SGOT: 13, 100, 200, 300, 400, 500, 
#     18. ALBUMIN: 2.1, 3.0, 3.8, 4.5, 5.0, 6.0
#     19. PROTIME: 10, 20, 30, 40, 50, 60, 70, 80, 90
#     20. HISTOLOGY: no, yes

# In[ ]:


# Getting the Column Names

hepatitis_df.columns


# In[ ]:


# Getting all the column which are numeric and provided info in description txt file so we can change into float values

numColumns = ['AGE','BILIRUBIN','ALK PHOSPHATE','SGOT','ALBUMIN','PROTIME']


# In[ ]:


# Converting string columns into float values

hepatitis_df["BILIRUBIN"] = hepatitis_df["BILIRUBIN"].astype(float)
hepatitis_df["BILIRUBIN"].head()


# In[ ]:


# Converting string columns into float values

hepatitis_df["ALK PHOSPHATE"] = hepatitis_df["ALK PHOSPHATE"].astype(float)
hepatitis_df["ALK PHOSPHATE"].head()


# In[ ]:


# Converting string columns into float values

hepatitis_df["SGOT"] = hepatitis_df["SGOT"].astype(float)
hepatitis_df["SGOT"].head()


# In[ ]:


# Converting string columns into float values

hepatitis_df["ALBUMIN"] = hepatitis_df["ALBUMIN"].astype(float)
hepatitis_df["ALBUMIN"].head()


# In[ ]:


# Converting string columns into float values

hepatitis_df["PROTIME"] = hepatitis_df["PROTIME"].astype(float)
hepatitis_df["PROTIME"].head()


# In[ ]:


# Checking the summary Statistics again to see if we are getting all the column now

hepatitis_df.describe()


# In[ ]:


# Getting the sum of all null values of particular column

hepatitis_df.isnull().sum()


# In[ ]:


# Fill Missing values with mean of the columns which have alot of nan values

hepatitis_df['BILIRUBIN'] = hepatitis_df['BILIRUBIN'].fillna(hepatitis_df['BILIRUBIN'].mean())
hepatitis_df['BILIRUBIN']


# In[ ]:


# Fill Missing values with mean of the columns which have alot of nan values

hepatitis_df['ALK PHOSPHATE'] = hepatitis_df['ALK PHOSPHATE'].fillna(hepatitis_df['ALK PHOSPHATE'].mean())
hepatitis_df['ALK PHOSPHATE']


# In[ ]:


# Fill Missing values with mean of the columns which have alot of nan values

hepatitis_df['SGOT'] = hepatitis_df['SGOT'].fillna(hepatitis_df['SGOT'].mean())
hepatitis_df['SGOT']


# In[ ]:


# Fill Missing values with mean of the columns which have alot of nan values

hepatitis_df['ALBUMIN'] = hepatitis_df['ALBUMIN'].fillna(hepatitis_df['ALBUMIN'].mean())
hepatitis_df['ALBUMIN']


# In[ ]:


# Fill Missing values with mean of the columns which have alot of nan values

hepatitis_df['PROTIME'] = hepatitis_df['PROTIME'].fillna(hepatitis_df['PROTIME'].mean())
hepatitis_df['PROTIME']


# In[ ]:


# Checking again now how many columns are still have nan values. There are stirng columns

hepatitis_df.isnull().sum()


# In[ ]:


# Replacing all the categorical column values with mode imputation

hepatitis_df = hepatitis_df.fillna(hepatitis_df.mode().iloc[0])


# In[ ]:


# Checking for null values again. Now dataset is preety clean

hepatitis_df.isnull().sum()


# In[ ]:


# Data types of each column

hepatitis_df.info()


# In[ ]:


# Converting whole dataset into float

hepatitis_df = hepatitis_df.astype(float)


# In[ ]:


# Varify if dataset is fully converted

hepatitis_df.info()


# In[ ]:


# Selecting all the features initially for our X

Feature = hepatitis_df.iloc[:, [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]]


# In[ ]:


X = Feature
X.head()


# In[ ]:


# Data Normalization for getting better and faster results

from sklearn import preprocessing
X = preprocessing.StandardScaler().fit(X).transform(X)
X[0:5]


# In[ ]:


# Selecting the output value (y)

output = hepatitis_df.iloc[:, [0]]


# In[ ]:


y = output
y.head()


# In[ ]:


# Converting the label into int value

y = y.astype(int)


# In[ ]:


y = y['Class'].values
y[0:5]


# # Applying KNN to Hepatitis Dataset

# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import jaccard_score
from sklearn import metrics

X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size = 0.25, random_state=0)
print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)


# In[ ]:


# Applying KNN on differnt values of K

k = 1
for i in range (20):
    knn = KNeighborsClassifier(n_neighbors = k).fit(X_train,Y_train)
    yhat = knn.predict(X_test)
    print ("KNN with k = " + str(k))
    print("Accuracy on train data with k = " + str (k) + " : ", metrics.accuracy_score(Y_train,knn.predict(X_train)) * 100)
    print("Accuracy on test data with  k = " + str(k) + " : ", metrics.accuracy_score(Y_test, yhat) * 100)
    print ()
    k+=1
    


# ## The best accuracy is on K = 3 with Train Accuracy = 93.04% & Test Accuracy = 87.17%

# # Applying Naive Bayes on Hepatitus Dataset

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0) 


# In[ ]:


# Bernaouli Naive Bayes

from sklearn.naive_bayes import BernoulliNB 
gnb = BernoulliNB() 
gnb.fit(X_train, y_train) 


# In[ ]:


y_pred = gnb.predict(X_test) 
y_trainPred = gnb.predict(X_train) 
  

from sklearn import metrics
print("Bernoulli Naive Bayes model accuracy on test set     :", metrics.accuracy_score(y_test, y_pred)*100)


# In[ ]:


# Gaussian Naive Bayes


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0) 
from sklearn.naive_bayes import GaussianNB 
gnb = GaussianNB() 
gnb.fit(X_train, y_train) 
y_pred = gnb.predict(X_test) 
y_trainPred = gnb.predict(X_train) 


print("Gaussian Naive Bayes model accuracy on test set     :", metrics.accuracy_score(y_test, y_pred)*100)


# ## The best accuracy is found on Bernoulli Naive Bayes with Test Acc = 83%

# In[ ]:


# Plotting Correlation Matrix to see if some features are dominating on other features and eradicating uselesss features

plt.subplots(figsize=(15, 15))
plt.title('Correlation Matrix')
sns.heatmap(Feature.astype(float).corr(), square=True, annot=True)


# In[ ]:


# Removing Problematic Features with negative correlation values i.e col# 2, 13, 14, 16, 19

Feature = hepatitis_df.iloc[:, [1,3,4,5,6,7,8,9,10,11,12,15,17,18]]


# In[ ]:


# Selecting our new features 

X = Feature


# In[ ]:


# Applying normalization

from sklearn import preprocessing
X = preprocessing.StandardScaler().fit(X).transform(X)
X[0:5]


# In[ ]:


# Splitting data again

X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size = 0.25, random_state=0)
print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)


# In[ ]:


# Applying KNN on differnt values of K

k = 1
for i in range (20):
    knn = KNeighborsClassifier(n_neighbors = k).fit(X_train,Y_train)
    yhat = knn.predict(X_test)
    print ("KNN with k = " + str(k))
    print("Accuracy on train data with k = " + str (k) + " : ", metrics.accuracy_score(Y_train,knn.predict(X_train)) * 100)
    print("Accuracy on test data with  k = " + str(k) + " : ", metrics.accuracy_score(Y_test, yhat) * 100)
    print ()
    k+=1
    


# # Now we are getting maximum accuracy of 92% but model is overfitting slightly

# In[ ]:


# Confusion Matrix After Feature Improvement

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

matrix = classification_report(Y_test,yhat)
print('Classification report : \n',matrix)


# In[ ]:





# In[ ]:





# # Liver-Disorder Dataset 
# 

# In[ ]:


# Reading dataset

liver_df = pd.read_csv("Datasets/liver_disorder_data.txt",delimiter=',')


# In[ ]:


liver_df.head()


# #  Attribute information:
#    1. mcv	mean corpuscular volume
#    2. alkphos	alkaline phosphotase
#    3. sgpt	alamine aminotransferase
#    4. sgot 	aspartate aminotransferase
#    5. gammagt	gamma-glutamyl transpeptidase
#    6. drinks	number of half-pint equivalents of alcoholic beverages drunk per day
#    7. selector  field used to split data into two sets
# 

# In[ ]:


# Giving the column names as given in text file

liver_df.columns = [
                        'mcv',
                        'alkphos',
                        'sgpt',
                        'sgot',
                        'gammagt',
                        'drinks',
                        'selector'
                    ]


# In[ ]:


liver_df.head(5)


# In[ ]:


liver_df.tail(5)


# In[ ]:


liver_df.shape


# In[ ]:


# Replacing ? with nan

liver_df.replace("?", np.nan, inplace = True)


# In[ ]:


liver_df.isnull().mean()


# In[ ]:


liver_df.isnull().sum()


# In[ ]:


liver_df.info()


# In[ ]:


liver_df = liver_df.astype(float)


# In[ ]:


liver_df = liver_df.fillna(liver_df.mean())


# In[ ]:


liver_df.describe()


# In[ ]:


liver_df.isin({0.0}).sum()


# In[ ]:


liver_df['selector'] = liver_df['selector'].astype(int)


# In[ ]:


Feature = liver_df.iloc[:, [0,1,2,3,4,5]]


# In[ ]:


X = liver_df.iloc[:, [0,1,2,3,4,5]]
X.head()


# In[ ]:


y = liver_df['selector'].values
y[0:5]


# In[ ]:


from sklearn import preprocessing
X = preprocessing.StandardScaler().fit(X).transform(X)
X[0:5]


# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size = 0.30, random_state=1)
print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)


# In[ ]:


k = 1
for i in range (20):
    knn = KNeighborsClassifier(n_neighbors = k).fit(X_train,Y_train)
    yhat = knn.predict(X_test)
    print ("KNN With k = " + str(k))
    print("Accuracy on train data with k = " + str (k) + " : ", metrics.accuracy_score(Y_train,knn.predict(X_train)) * 100)
    print("Accuracy on test data with  k = " + str(k) + " : ", metrics.accuracy_score(Y_test, yhat) * 100)
    print ()
    k+=1
    


# In[ ]:


# Confusion Matrix Before Feature Improvement

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

matrix = classification_report(Y_test,yhat)
print('Classification report : \n',matrix)


# # The best accuracy is on K = 19 with Accuracy = 70.19%

# # Naive Bayes Classifier

# In[ ]:



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0) 
from sklearn.naive_bayes import GaussianNB 
gnb = GaussianNB() 
gnb.fit(X_train, y_train) 
y_pred = gnb.predict(X_test)   

from sklearn import metrics
print("Gaussian Naive Bayes model accuracy on test set     :", metrics.accuracy_score(y_test, y_pred)*100)


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0) 
from sklearn.naive_bayes import BernoulliNB 
gnb = BernoulliNB() 
gnb.fit(X_train, y_train) 
y_pred = gnb.predict(X_test)   

from sklearn import metrics
print("Bernoulli Naive Bayes model accuracy on test set     :", metrics.accuracy_score(y_test, y_pred)*100)


# In[ ]:


plt.subplots(figsize=(10, 10))
plt.title('Correlation Matrix')
sns.heatmap(Feature.astype(float).corr(), square=True, annot=True)


# ### Selected those features which are highly correlated

# In[ ]:


# Selecting sgpt, sgot, gammagt and drinks due to highly correlated results

X = liver_df.iloc[:, [2,3,4,5]]
X.head()


# In[ ]:


# Applying normalization

from sklearn import preprocessing
X = preprocessing.StandardScaler().fit(X).transform(X)
X[0:5]


# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size = 0.30, random_state=1)
print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)


# In[ ]:


predd = []
k = 1
for i in range (20):
    knn = KNeighborsClassifier(n_neighbors = k).fit(X_train,Y_train)
    yhat = knn.predict(X_test)
    print ("KNN With k = " + str(k))
    print("Accuracy on train data with k = " + str (k) + " : ", metrics.accuracy_score(Y_train,knn.predict(X_train)) * 100)
    print("Accuracy on test data with  k = " + str(k) + " : ", metrics.accuracy_score(Y_test, yhat) * 100)
    print ()
    k+=1
    


# In[ ]:


# Confusion Matrix After Feature Improvement

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

matrix = classification_report(Y_test,yhat)
print('Classification report : \n',matrix)


# In[ ]:


# Getting confusion Matrix

def perf_measure(y_actual, y_hat):
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(len(y_hat)): 
        if y_actual[i]==y_hat[i]==1:  
            TP += 1
        if y_hat[i]==1 and y_actual[i]!=y_hat[i]:
            FP += 1
        if y_actual[i]==y_hat[i]==2:
            TN += 1
        if y_hat[i]==0 and y_actual[i]!=y_hat[i]:
            FN += 1

    return(TP, FP, TN, FN)

tp, fp, tn, fn = perf_measure(Y_test, yhat)

print ("True Positive  : " + str(tp))
print ("False Positive : " + str(fp))
print ("True Nagative  : "  + str(tn))
print ("False Negative : " + str(fn))


# In[ ]:


# Cross Validaiton With Bin Size of 5 and tested on different sizes

from sklearn.model_selection import cross_val_score

k = 1

for i in range (20):
    print ("K = " + str(k))
    knn_cv = KNeighborsClassifier(n_neighbors = k)

    cv_scores = cross_val_score(knn_cv, X, y, cv = 5)
    print(cv_scores)
    print('Cross Validation Scores Mean:{}'.format(np.mean(cv_scores)))
    k = k + 1
    print ()


# # Lung-Cancer Data

# In[ ]:


lungCancer_df = pd.read_csv("Datasets/lung_cancer_data.txt", delimiter=',')


# In[ ]:


lungCancer_df.head(5)


# In[ ]:


lungCancer_df.shape


# ## No information about columns are given so col1, col2 .. names are assigned

# In[ ]:


# Giving column Values

lungCancer_df.columns = [
                        'col1','col2','col3','col4','col5','col6','col7','col8','col9','col10',
                        'col11','col12','col13','col14','col15','col16','col17','col18','col19','col20',
                        'col21','col22','col23','col24','col52','col26','col27','col28','col29','col30',
                        'col31','col32','col33','col34','col35','col36','col37','col38','col39','col40',
                        'col41','col42','col43','col44','col45','col46','col47','col48','col49','col50',
                        'col51','col52','col53','col54','col55','col56','col57'
                    ]


# In[ ]:


lungCancer_df.head()


# In[ ]:


lungCancer_df.replace("?", np.nan, inplace = True)


# In[ ]:


lungCancer_df.isnull().sum()


# In[ ]:


lungCancer_df.info()


# In[ ]:


lungCancer_df = lungCancer_df.astype(float)


# In[ ]:


# Removing all null values

lungCancer_df['col5'] = lungCancer_df.fillna(lungCancer_df['col5'].mode())
lungCancer_df['col2'] = lungCancer_df.fillna(lungCancer_df['col2'].mode())
lungCancer_df['col6'] = lungCancer_df.fillna(lungCancer_df['col6'].mode())
lungCancer_df['col12'] = lungCancer_df.fillna(lungCancer_df['col12'].mode())
lungCancer_df['col29'] = lungCancer_df.fillna(lungCancer_df['col29'].mode())
lungCancer_df['col30'] = lungCancer_df.fillna(lungCancer_df['col30'].mode())
lungCancer_df['col39'] = lungCancer_df.fillna(lungCancer_df['col39'].mode())
lungCancer_df['col42'] = lungCancer_df.fillna(lungCancer_df['col42'].mode())
lungCancer_df['col48'] = lungCancer_df.fillna(lungCancer_df['col48'].mode())
lungCancer_df['col51'] = lungCancer_df.fillna(lungCancer_df['col51'].mode())
lungCancer_df['col53'] = lungCancer_df.fillna(lungCancer_df['col53'].mode())


# In[ ]:


lungCancer_df.isnull().sum()


# In[ ]:


# Selecting all features initially and then select only those which are good

Feature = lungCancer_df.loc[:, lungCancer_df.columns != 'col57']


# In[ ]:


X = Feature


# In[ ]:


y = lungCancer_df['col57'].values


# In[ ]:


from sklearn import preprocessing
X = preprocessing.StandardScaler().fit(X).transform(X)
X[0:2]


# In[ ]:



X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size = 0.35, random_state=0)
print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)


# In[ ]:


# KNN with different k values

k = 1
for i in range (20):
    knn = KNeighborsClassifier(n_neighbors = k).fit(X_train,Y_train)
    yhat = knn.predict(X_test)
    print ("KNN With k = " + str(k))
    print("Accuracy on train data with k = " + str (k) + " : ", metrics.accuracy_score(Y_train,knn.predict(X_train)) * 100)
    print("Accuracy on test data with  k = " + str(k) + " : ", metrics.accuracy_score(Y_test, yhat) * 100)
    print ()
    k+=1


# ## KNN is giving high accuracy of 72%  which is very biased because we have only 31 rows

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, random_state=0) 
from sklearn.naive_bayes import BernoulliNB 
gnb = BernoulliNB() 
gnb.fit(X_train, y_train) 
y_pred = gnb.predict(X_test)   

from sklearn import metrics
print("Bernoulli Naive Bayes model accuracy on test set     :", metrics.accuracy_score(y_test, y_pred)*100)


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, random_state=0) 
from sklearn.naive_bayes import GaussianNB 
gnb = GaussianNB() 
gnb.fit(X_train, y_train) 
y_pred = gnb.predict(X_test)   

from sklearn import metrics
print("Gaussian Naive Bayes model accuracy on test set     :", metrics.accuracy_score(y_test, y_pred)*100)


# In[ ]:


# Correlation Matrix

plt.subplots(figsize=(30, 30))
plt.title('Correlation Matrix')
sns.heatmap(Feature.astype(float).corr(), square=True, annot=True)


# ### There is no Conclusive Relation among different columns so we can go with all features selected

# In[ ]:





# In[ ]:




