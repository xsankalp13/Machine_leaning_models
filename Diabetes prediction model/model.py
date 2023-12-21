import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

#loading the dataset to a pandas dataframe
diabetes_dataset = pd.read_csv('./data/diabetes.csv')

#seprating data and lables
data = diabetes_dataset.drop(columns='Outcome',axis=1)
lables = diabetes_dataset['Outcome']

scaler = StandardScaler()
scaler.fit(data)
standardized_data = scaler.transform(data)
data = standardized_data

data_train, data_test, lables_train, lables_test = train_test_split(data,lables,test_size=0.2,stratify=lables,random_state=2)

classifier = svm.SVC(kernel = 'linear')

#traing the SVM classifier
classifier.fit(data_train,lables_train)

#making predictive system
input_data = (9,171,110,24,240,45.4,0.721,54)
#changing data to numpy array
input_data_array = np.asanyarray(input_data)
#reshape the array
input_data_shaped = input_data_array.reshape(1,-1)
#standardize the input data
input_data_std = scaler.transform(input_data_shaped)

prediction = classifier.predict(input_data_std)

if prediction == 0:
  print('No, she is not Diabatic.')
else:
  print("Yes, she is Diabatic.")