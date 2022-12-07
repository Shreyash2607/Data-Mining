import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import load_iris
iris = load_iris()

X = iris.data[:, :4]  # we only take the first  features.
y = iris.target

from matplotlib import pyplot as plt

# The indices of the features that we are plotting
x_index = 0
y_index = 1

# this formatter will label the colorbar with the correct target names
formatter = plt.FuncFormatter(lambda i, *args: iris.target_names[int(i)])

plt.figure(figsize=(8, 8))
plt.scatter(iris.data[:, x_index], iris.data[:, y_index], c=iris.target)
plt.colorbar(ticks=[0, 1, 2], format=formatter)
plt.xlabel(iris.feature_names[x_index])
plt.ylabel(iris.feature_names[y_index])
plt.title('2D view of IRIS')

plt.tight_layout()
plt.show()

### SPLITTING INTO TRAINING AND TEST SETS
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25,random_state=28)

### NORMALIZTION / FEATURE SCALING
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

### WE WILL FIT THE THE CLASSIFIER TO THE TRAINING SET
naiveClassifier=GaussianNB()
naiveClassifier.fit(X_train,y_train)

y_pred = naiveClassifier.predict(X_test)

#Keeping the actual and predicted value side by side
y_compare = np.vstack((y_test,y_pred)).T
#Actual->LEFT
#predicted->RIGHT
#Number of values to be print
y_compare[:20,:]

plt.scatter(y_test, y_pred)
plt.xlabel("ACTUAL VALUES")
plt.ylabel("PREDICTED")
plt.title("Actual and predicted value")
plt.show()

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)
# find total number of rows and
# columns in list

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

from tkinter import messagebox
root= Tk()
root.title('confusion_matrix')
root.geometry("700x300")
total_rows = len(cm)
total_columns = len(cm[0])
txt="Naive Bayes Clasification IRIS"
CMatrixButton = Button(root,text='Display confusion matrix', command=lambda: Table(cm,total_rows,total_columns,txt), bg='green', fg='white', font=('helvetica', 12, 'bold'))
CMatrixButton.place(x=50,y=150)

#Accuracy
from sklearn import metrics
from sklearn.metrics import accuracy_score

def display_acc(y_test, y_pred):
    messagebox.showinfo('Accuracy Of Model','Accuracy of the Naive Bayes Clasification is : {0:0.4f}'. format(accuracy_score(y_test, y_pred)))

#print accuracy
ACC_Button = Button(root,text='Display Accuracy of Model', command=lambda: display_acc(y_test, y_pred), bg='green', fg='white', font=('helvetica', 12, 'bold'))
ACC_Button.place(x=50,y=200)


from sklearn.metrics import classification_report
from  sklearn.metrics import precision_recall_fscore_support

def display_repo(y_test,y_pred):
    messagebox.showinfo ('Naive Bayes Clasification report ', classification_report(y_test, y_pred))


CreportButton = Button(root,text='Display classification report', command=lambda: display_repo(y_test,y_pred), bg='green', fg='white', font=('helvetica', 12, 'bold'))
CreportButton.place(x=50,y=250)
#finding accuracy from the confusion matrix.
a = cm.shape
correctPrediction = 0
falsePrediction = 0

for row in range(a[0]):
    for c in range(a[1]):
        if row == c:
            correctPrediction +=cm[row,c]
        else:
            falsePrediction += cm[row,c]
print('Correct predictions: ', correctPrediction)
print('False predictions', falsePrediction)
print ('\n\nAccuracy of the Naive Bayes Clasification is: ', correctPrediction/(cm.sum()))

#msg="Correct predictions:", correctPrediction, " False predictions",falsePrediction,"\nAccuracy of the Naive Bayes Clasification is: "+ correctPrediction/(cm.sum())

#messagebox.showinfo('Correct predictions: ', correctPrediction)
#messagebox.showinfo('False predictions', falsePrediction)
#messagebox.showinfo ('Accuracy of the Naive Bayes Clasification is: ', correctPrediction/(cm.sum()))
import seaborn as sns
import itertools
import matplotlib.colors as colors

sns.set()
#Load the data set
iris = sns.load_dataset("iris")
iris = iris.rename(index = str, columns = {'sepal_length':'1_sepal_length','sepal_width':'2_sepal_width', 'petal_length':'3_petal_length', 'petal_width':'4_petal_width'})

from sklearn.naive_bayes import GaussianNB
df1 = iris[["1_sepal_length", "2_sepal_width",'species']]
#Setup X and y data
X_data = df1.iloc[:,0:2]
y_labels = df1.iloc[:,2].replace({'setosa':0,'versicolor':1,'virginica':2}).copy()

#Fit model
model_sk = GaussianNB(priors = None)
model_sk.fit(X_data,y_labels)


# Our 2-dimensional classifier will be over variables X and Y
N = 100
X = np.linspace(4, 8, N)
Y = np.linspace(1.5, 5, N)
X, Y = np.meshgrid(X, Y)

#fig = plt.figure(figsize = (10,10))
#ax = fig.gca()
color_list = ['Coral','gold','navy']
my_norm = colors.Normalize(vmin=-1.,vmax=1.)

g = sns.FacetGrid(iris, hue="species", height=5 , palette = 'colorblind') .map(plt.scatter, "1_sepal_length", "2_sepal_width",)  .add_legend()
my_ax = g.ax


#Computing the predicted class function for each value on the grid
zz = np.array(  [model_sk.predict( [[xx,yy]])[0] for xx, yy in zip(np.ravel(X), np.ravel(Y)) ] )


#Reshaping the predicted class into the meshgrid shape
Z = zz.reshape(X.shape)


#Plot the filled and boundary contours
my_ax.contourf( X, Y, Z, 2, alpha = .3, colors = ('magenta','yellow','black'))
my_ax.contour( X, Y, Z, 2, alpha = 1, colors = ('orange','lime','cyan'))

# Addd axis and title
my_ax.set_xlabel('Sepal length')
my_ax.set_ylabel('Sepal width')
my_ax.set_title('Gaussian Naive Bayes decision boundaries')

plt.show()

root.mainloop()