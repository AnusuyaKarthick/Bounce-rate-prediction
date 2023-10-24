# -*- coding: utf-8 -*-


#importing necessary libraries
import pyodbc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#connecting sql and python using pyodbc library
conn = pyodbc.connect('Driver={SQL Server};'
                      'Server=DESKTOP-25S22SP\SQLEXPRESS;'
                      'Database=anu;'
                      'Trusted_Connection=yes;')

#importing data from sql table to python dataframe
df = pd.read_sql_query('SELECT * FROM [dbo].[Projectfinaldata]', conn)

#Display maximum columns
pd.set_option('display.max_columns',None)
print(df)

#print type of data
print(type(df))

#no. of rows 14218 and columns 14 in df
df.shape

#First five rows in the dataset
df.head()

#Last five rows in the dataset
df.tail()

#To view the categorical and numerical columns and its datatypes in the dataset
df.info()  #6 numerical and 7 categorical columns( 1 time object)

#To see the column name 
df.columns
# Select unique values from the data
df['Dept'].unique()
df['Formulation'].unique() #None values are present
df['DrugName'].unique()
df['SubCat'].unique()
df["SubCat1"].unique
df['Specialisation'].unique()
df["Quantity"].unique()
df["ReturnQuantity"].unique
#changing the dependent variable typeofsales column position to last
data1= df.drop('Typeofsales',axis=1)
data2=df.drop(df.iloc[:,1:],axis=1)
#concat the dataset
df = pd.concat([data1, data2], axis=1)

#Data Preprocessing
# Data Cleaning
#Dropping the columns not necessary for analysis
df=df.drop(['Patient_ID','Dateofbill'],axis=1)


#To find the statistical property of data
df.describe() 

#descriptive analysis
df.mean()
df.median()  #outliers are present as mean and median are not same.

#Handling missing values
#df = df.replace(to_replace='None', value=np.nan) # replacing none to nan and imputing na values
#df.dropna(inplace=True)
# checking any null values
df.isnull().sum()
#percentage of missing values
df.isnull().sum() * 100 / len(df) #considering the percent of missing values as less drop the missing values
#imputation -mode , drop missing values in formulation
df['DrugName'] = df['DrugName'].fillna(df['DrugName'].mode()[0])
df['SubCat'] = df['SubCat'].fillna(df['SubCat'].mode()[0])
df['SubCat1'] = df['SubCat1'].fillna(df['SubCat1'].mode()[0])
df.dropna(inplace=True)
df.isnull().sum()

# Visualisation 'Graphical Representation'

#To check imbalance 
df['Typeofsales'].value_counts()  
#2 classes in target feature - sale and return

plt.pie(df.Typeofsales.value_counts(), labels = ['Sale', 'Return',], autopct='%1.f%%', pctdistance=0.5)
plt.title('Bounce rate')  # imbalanced dataset


sns.countplot(df['Typeofsales'])
# given data is imbalanced dataset.

#boxplot - checking presence of outliers
sns.boxplot(data=df['Final_Cost'])

plt.figure(figsize=(10, 6))
sns.boxplot(data=df)
plt.show()    #Quantity and Return Quantity - Discrete data type

#Final sales, final cost and ReturnMRP - Continuous data has outliers

#creating winsorization techniques to handle outliers
from feature_engine.outliers import Winsorizer
iqr_winsor = Winsorizer(capping_method='iqr', tail='both',fold=1)

#handling outliers
df[['Final_Cost']] = iqr_winsor.fit_transform(df[['Final_Cost']])
df[['Final_Sales']] = iqr_winsor.fit_transform(df[['Final_Sales']])

#f[['RtnMRP']] = iqr_winsor.fit_transform(df[['RtnMRP']])


# boxplot for checking outliers after winsorization
plt.figure(figsize=(10, 6))
sns.boxplot(data=df)
plt.show()   

#checking duplicates : 118 duplicates are present
duplicates=df.duplicated()
duplicates
sum(duplicates)

#Removing duplicates
df.drop_duplicates(inplace=True)

#Label encoding
from sklearn.preprocessing import LabelEncoder
categories = ['Specialisation', 'Dept','Formulation', 'DrugName','SubCat', 'SubCat1', 'Typeofsales']
# Encode Categorical Columns
le = LabelEncoder()
df[categories] = df[categories].apply(le.fit_transform)
df 

#for finding relevant features for analysis
#Correlation 
corrMatrix = df.corr()
corrMatrix

# Correlation between different variables
corr = df.corr()
# Set up the matplotlib plot configuration
f, ax = plt.subplots(figsize=(30, 10))
# Generate a mask for upper traingle
mask = np.triu(np.ones_like(corr, dtype=bool))
# Configure a custom diverging colormap
cmap = sns.diverging_palette(230, 20, as_cmap=True)
# Draw the heatmap
sns.heatmap(corr, annot=True, mask = mask, cmap=cmap,robust=True,cbar=False,annot_kws={'size':16})

#target variable is dependent on finalsales,Dept,Quantity,Finalcost
#negatively correlated with formulation,drugname,subcat1

#separating dependent and independent columns 
X = df.iloc[:,0:11]  #independent columns
y = df.iloc[:,-1]    #target column


'''
#Feature Engineering with chi-square test
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
test = SelectKBest(score_func=chi2, k=10)
fit = test.fit(X, y)
X_new=test.fit_transform(X, y)
fit.scores_'''

#split into train and test
from sklearn.model_selection import train_test_split
# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=30) # 70% training and 30% test
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
#Create a Gaussian Classifier
clf=RandomForestClassifier(n_estimators=100,max_depth=3)

#Train the model using the training sets y_pred=clf.predict(X_test)
clf.fit(X_train,y_train)
#Predict the response for test dataset
y_pred=clf.predict(X_test)
# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_train,clf.predict(X_train) ))
# Model Accuracy
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))



#Desicion tree Classification Algorithm
#Load libraries
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from sklearn.metrics import accuracy_score,confusion_matrix
#separating dependent and independent columns 
X = df.iloc[:,0:11]  #independent columns
y = df.iloc[:,-1]    #target column
# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=30) 
# 70% training and 30% test

# Model building for Decision Tree
clf1 = DecisionTreeClassifier()
clf1.fit(X_train,y_train)
y_train_pred = clf.predict(X_train)
y_test_pred = clf.predict(X_test)

print(f'Train score {accuracy_score(y_train_pred,y_train)}') # 
print(f'Test score {accuracy_score(y_test_pred,y_test)}')  #  
# confusion matrix for performance metrics
cm = confusion_matrix(y_test, y_test_pred)
cm

p=pd.crosstab(y_test, y_test_pred, rownames=['Actual'], colnames=['Predictions'])

sns.heatmap(p,annot=True,fmt=".1f",robust=True,cbar=False,annot_kws={'size':16})


# Classification Report
from sklearn.metrics import classification_report
print(classification_report(y_test, y_test_pred))

import pickle
# open the pickle file in writebyte mode
file = open("model.pkl",'wb')
#dump information to that file
pickle.dump(clf1, file)
file.close()
# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
print(model.predict([[46,0,1,0,55.406,59.26,0,0,737,14,20]]))

y_pred_pickle = model.predict(X_test)
print(f'Test score {accuracy_score(y_pred_pickle,y_test)}')
