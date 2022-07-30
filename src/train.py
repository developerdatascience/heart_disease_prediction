import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import pickle



# Data Collection and Data Processing
heart_data = pd.read_csv("input/heart_disease_data.csv")

print(heart_data.head())

#Checking out for missing value
print(heart_data.isnull().sum())

#Checking out the distribution for target variables
print(heart_data['target'].value_counts())  # 1 -> At Risk , 0 -> Not At Risk


# Splitting the Features and target

X = heart_data.drop(columns='target', axis=1)
y = heart_data['target']


#Splitting the data into training and test data
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=3)


#Model Training :- Logistic Regression

#Training the Logistic regression model with training data
model = LogisticRegression()

model.fit(X_train, Y_train)

X_train_prediction = model.predict(X_train)
train_accuracy_score = accuracy_score(X_train_prediction,Y_train)

print("Accuracy score on training data: ", train_accuracy_score)

#Testing the Logistic Regression model with test data
X_test_prediction = model.predict(X_test)
test_accuracy_score = accuracy_score(X_test_prediction, Y_test)

print("Accuracy score on test data: ", test_accuracy_score)

#Building Predictive System

input_data = (57,1,0,150,276,0,0,112,1,0.6,1,1,1)

# Change the input data to numpy array
input_data_as_numpy_array = np.asarray(input_data)

# Reshaping the input_data_as_numpy_array as we are predicting for one instance
input_data_reshape = input_data_as_numpy_array.reshape(1, -1)

prediction = model.predict(input_data_reshape)
print(prediction)


if prediction[0] == 0:
    print("Person does not have heart disease")
else:
    print("Person have heart disease")


#Creatig a dump of model

pickle.dump(model, open("logisticmodel.pkl", "wb"))

pickle_model = pickle.load(open("logisticmodel.pkl", "rb"))

print(pickle_model.predict(input_data_reshape)[0])

