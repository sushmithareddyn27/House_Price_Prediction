'''step1:- Importing libraries'''

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing

'''Step2:- Loading the dataset into variable '''

housing=fetch_california_housing()
type(housing)

'''step 3:- Data Preparation'''

dataset=pd.DataFrame(housing.data,columns=housing.feature_names)
#print(dataset.head()) 
#to print first 5 elements of dataframe,tail for last 5 reports'''
dataset["Price"]=housing.target
#dataset.info() helps to get the information of all the variables 
#dataset.describe() give all the ststical information about the dataset
#dataset.isnull().sum() to check the null values in the dataset

'''step 4:- Exploratory data analysis'''

#print(dataset.corr()) to check the correlation between the variables
#sns.pairplot(dataset) to check the relationship between the variables
#plt.show()
#Boxplot to check the outliers in the dataset
#sns.boxplot(data=dataset)
#plt.show()
#split data into dependent and independent variables
x=dataset.iloc[:,:-1] #all rows and columns except last column
y=dataset.iloc[:,-1]#all rows and only last column
# split data into training and testing data

from sklearn.model_selection import train_test_split;
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,train_size=0.7,random_state=42)
#normalization of given datapoints
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
x_train_norm=scaler.fit_transform(x_train)
#then u can do boxplot to observe difference
x_test_norm=scaler.transform(x_test)

''' step 5:- Model training'''

from sklearn.linear_model import LinearRegression
regression=LinearRegression()
regression.fit(x_train_norm,y_train)

print(regression.coef_)
print(regression.intercept_)

'''step 6:- Model testing'''

reg_pred= regression.predict(x_test_norm)

#calculate resuidal or error 
residuals=y_test-reg_pred

'''step 7:- Model evaluation'''
#lower the value of mean absolute error and mean squared error better the model is, closer the r2 score to 1 better the model is
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
print("Mean Absolute Error:",mean_absolute_error(y_test,reg_pred))
print("Mean Squared Error:",mean_squared_error(y_test,reg_pred))
print("R2 Score:",r2_score(y_test,reg_pred))  

#step 7 :- save the model for future use
import pickle
#wb means write binary mode, rb means read binary mode
pickle.dump(regression,open("house_price_prediction_model.pkl","wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))
#to load the model
model=pickle.load(open("house_price_prediction_model.pkl","rb"))
scaler=pickle.load(open("scaler.pkl","rb"))

model.predict(scaler.transform(housing.data[0].reshape(1,-1)))
#we use reshape to convert 1D array to 2D array because scaler expects 2D array as input
