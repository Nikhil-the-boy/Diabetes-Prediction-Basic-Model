#                                                   DIABETES PREDICTION AI

# Importing the dependencies
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score 

#____________________________________________________________________________________________

# Loading the dataset
diabetes_dataset = pd.read_csv(r'C:\Users\visha\Downloads\diabetes.csv')

# print(diabetes_dataset.head())

# No of rows and columns of dataset -->
# print(diabetes_dataset.shape)

# Getting the statistical data of the dataset
# print(diabetes_dataset.describe())

# Counting the No. of values of Outcome column(As it is the result)
# print(diabetes_dataset['Outcome'].value_counts())

# print(diabetes_dataset.groupby('Outcome').mean())

# Seperating the data and labels
x = diabetes_dataset.drop(columns='Outcome', axis=1)
y = diabetes_dataset['Outcome']

# print(y)

# __________________________________________________________________________________________________________

#           Now I am doing Data Standardization

scaler = StandardScaler()
scaler.fit(x)
Standardized_data = scaler.transform(x)

x = Standardized_data
y = diabetes_dataset['Outcome']

#____________________________________________________________________________________________________________

#           Now I am doing train, test, split

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2, stratify=y, random_state=2)
# print(x.shape,x_train.shape,x_test.shape)

#_____________________________________________________________________________________________________________

#           Now I am training the model

classifier = svm.SVC(kernel='linear')

# Training the support vector machine classifier
classifier.fit(x_train, y_train)

#           Accuracy Score on the training data
x_train_prediction = classifier.predict(x_train)
training_data_accuracy = accuracy_score(x_train_prediction, y_train)

print("Accuracy score of the training data : ", training_data_accuracy)

#           Accuracy score on the test data
x_test_prediction = classifier.predict(x_test)
test_data_accuracy = accuracy_score(x_test_prediction, y_test)

print("Accuracy score of the test data : ", test_data_accuracy)



#______________________________________________________________________________________________________________

#            Making predictive system
input_data = (4,110,92,0,0,37.6,0.191,30)

# Chnaging the input_data to numpy array
input_data_as_numpy_array = np.asarray(input_data)

# Reshape the array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

# Standardized the input data
std_data = scaler.transform(input_data_reshaped)

prediction = classifier.predict(std_data)
print(prediction)