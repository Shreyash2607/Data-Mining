import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Assign colum names to the dataset
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'Class']

# Read dataset to pandas dataframe
dataset = pd.read_csv('iris.data', names=names)

#The next step is to split our dataset into its attributes and labels. 
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

#Train
#splits the dataset into 80% train data and 20% test data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)

X_test = scaler.transform(X_test)
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(X_train, y_train)

#Make prediction
y_pred = classifier.predict(X_test)

#Evaluate
from sklearn.metrics import classification_report, confusion_matrix
cm=confusion_matrix(y_test, y_pred)
cr=classification_report(y_test, y_pred)


#To find the best value of K is to plot the graph 
#of K value and the corresponding error rate for the dataset
error = []

# Calculating error for K values between 1 and 40
for i in range(1, 40):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    pred_i = knn.predict(X_test)
    error.append(np.mean(pred_i != y_test))

plt.figure(figsize=(12, 6))
plt.plot(range(1, 40), error, color='red', linestyle='dashed', marker='o',
         markerfacecolor='green', markersize=10)
plt.title('Error Rate vs K Value')
plt.xlabel('K Value')
plt.ylabel('Mean Error')
plt.show()

#mean error is zero when the value of the K is between

class Table:
      
    def __init__(self,cm,total_rows,total_columns,txt):
        # setup new window
        #mat_window = Toplevel(root)
        #mat_window.title("naiveClassifier : Confusion Matrix :"+txt) 
        # code for creating table
        for i in range(total_rows):
            for j in range(total_columns):
                  
                self.e = Entry(root, width=20, fg='blue',
                               font=('Arial',16,'bold'))
                  
                self.e.grid(row=i, column=j)
                self.e.insert(END, cm[i][j])

from tkinter import *

root= Tk()
root.title('confusion_matrix')
root.geometry("700x300")
total_rows = len(cm)
total_columns = len(cm[0])
txt="kNN Clasification IRIS"
CMatrixButton = Button(root,text='Display confusion matrix', command=lambda: Table(cm,total_rows,total_columns,txt), bg='green', fg='white', font=('helvetica', 12, 'bold'))
CMatrixButton.place(x=50,y=150)


#Accuracy
from sklearn import metrics
from sklearn.metrics import accuracy_score

def display_acc(y_test, y_pred):
    messagebox.showinfo('Accuracy Of Model','Accuracy of the KNN Clasification for k= 5 is : {0:0.4f}'. format(accuracy_score(y_test, y_pred)))

#print accuracy
ACC_Button = Button(root,text='Display Accuracy of Model', command=lambda: display_acc(y_test, y_pred), bg='green', fg='white', font=('helvetica', 12, 'bold'))
ACC_Button.place(x=50,y=200)



def display_repo():
    messagebox.showinfo ('kNN Clasification report ', cr)


CreportButton = Button(root,text='Display classification report', command=lambda: display_repo(), bg='green', fg='white', font=('helvetica', 12, 'bold'))
CreportButton.place(x=50,y=250)
root.mainloop()




