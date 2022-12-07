import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

# Function importing Dataset
def importdata():
balance_data = pd.read_csv(
'https://archive.ics.uci.edu/ml/machine-learning-'+
'databases/balance-scale/balance-scale.data',
sep= ',', header = None)

# Printing the dataswet shape
print ("Dataset Length: ", len(balance_data))
print ("Dataset Shape: ", balance_data.shape)

# Printing the dataset obseravtions
print ("Dataset: ",balance_data.head())
return balance_data

# Function to split the dataset
def splitdataset(balance_data):

# Separating the target variable
X = balance_data.values[:, 1:5]
Y = balance_data.values[:, 0]

# Splitting the dataset into train and test
x-train, X-test, y-train, y-test = train_test_split(
X, Y, test-size = 0.3, random-state = 100)

return X, Y, x-train, X-test, y-train, y-test

# Function to perform training with giniIndex.
def train_using_gini(x-train, X-test, y-train):

# Creating the classifier object
clf_gini = DecisionTreeClassifier(criterion = "gini",
random-state = 100,max_depth=3, min_samples_leaf=5)

# Performing training
clf_gini.fit(x-train, y-train)
return clf_gini

# Function to perform training with entropy.
def tarin_using_entropy(x-train, X-test, y-train):

# Decision tree with entropy
clf_entropy = DecisionTreeClassifier(
criterion = "entropy", random-state = 100,
max_depth = 3, min_samples_leaf = 5)

# Performing training
clf_entropy.fit(x-train, y-train)
return clf_entropy

# Function to make predictions
def prediction(X-test, clf_object):

# Predicton on test with giniIndex
Y_prediction = clf_object.predict(X-test)
print("Predicted values:")
print(Y_prediction)
return Y_prediction

# Function to calculate accuracy
def cal_accuracy(y-test, Y_prediction):

print("Confusion Matrix: ",
confusion_matrix(y-test, Y_prediction))

print ("Accuracy : ",
accuracy_score(y-test,Y_prediction)*100)

print("Report : ",
classification_report(y-test, Y_prediction))

# Driver code
def main():

# Building Phase
data = importdata()
X, Y, x-train, X-test, y-train, y-test = splitdataset(data)
clf_gini = train_using_gini(x-train, X-test, y-train)
clf_entropy = tarin_using_entropy(x-train, X-test, y-train)

# Operational Phase
print("Results Using Gini Index:")

# Prediction using gini
Y_prediction_gini = prediction(X-test, clf_gini)
cal_accuracy(y-test, Y_prediction_gini)

print("Results Using Entropy:")
# Prediction using entropy
Y_prediction_entropy = prediction(X-test, clf_entropy)
cal_accuracy(y-test, Y_prediction_entropy)

# Calling main function
if __name__=="__main__":
main()