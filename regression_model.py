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
dataset["Price"]=housing.target

'''step 4:- Exploratory data analysis'''

x=dataset.iloc[:,:-1] 
y=dataset.iloc[:,-1]

from sklearn.model_selection import train_test_split;
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,train_size=0.7,random_state=42)
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
x_train_norm=scaler.fit_transform(x_train)
x_test_norm=scaler.transform(x_test)

''' step 5:- Model training'''

from sklearn.linear_model import LinearRegression
regression=LinearRegression()
regression.fit(x_train_norm,y_train)

print(regression.coef_)
print(regression.intercept_)

'''step 6:- Model testing'''

reg_pred= regression.predict(x_test_norm)
residuals=y_test-reg_pred

'''step 7:- Model evaluation'''
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
print("Mean Absolute Error:",mean_absolute_error(y_test,reg_pred))
print("Mean Squared Error:",mean_squared_error(y_test,reg_pred))
print("R2 Score:",r2_score(y_test,reg_pred))  

'''step 8 :- save the model for future use'''
import pickle

pickle.dump(regression,open("house_price_prediction_model.pkl","wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))
model=pickle.load(open("house_price_prediction_model.pkl","rb"))
scaler=pickle.load(open("scaler.pkl","rb"))

model.predict(scaler.transform(housing.data[0].reshape(1,-1)))
