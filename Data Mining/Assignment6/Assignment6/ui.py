# Import module
from tkinter import *
from tkinter import filedialog
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

# Create object
root = Tk()
root.title("Assignment 6")
root.configure(bg='palegreen')
User_input = Entry()



# def get_excel():
	# import_file_path = filedialog.askopenfilename()


# Adjust size
root.geometry( "2000x1000" )
# lbl = Label (root, text="Assignment No 6",  fg="black", font="none 20 bold")

# Change the label text
def show():
    
    dataset = clicked.get()
    calgo = classifier.get()
    if dataset=="Iris":
        data = load_iris()
        # print(data.data)
        if calgo=="Regression Classifier":
            IrisReg(data)
        elif calgo=="Naive Bayesian Classifier":
            IrisNaive(data)
        elif calgo=="k-nn Classifier":
            User_input.pack()
            # User_input.get()
            # print(User_input)
            # User_input.insert(0,"Enter the value of K")
           
            IrisKnn(data,int(User_input.get()))
            User_input.delete(0,20)
        else:
            print("Iris ANN Classifier")
    else:
        data = load_breast_cancer()
        print(data.data)
        if calgo=="Regression Classifier":
            print("Cancer Regression Classifier")
        elif calgo=="Naive Bayesian Classifier":
            print("Cancer Naive Bayesian Classifier")
        elif calgo=="k-nn Classifier":
            print("Cancer k-nn Classifier")
        else:
            print("Cancer ANN Classifier")
    # label.config( text = clicked.get() )
    # clabel.config( text = classifier.get())



def IrisReg(iris):
    X = iris.data[:, :2]  # we only take the first two features.
    Y = iris.target

    # Create an instance of Logistic Regression Classifier and fit the data.
    logreg = LogisticRegression(C=1e5)
    logreg.fit(X, Y)

    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = .02  # step size in the mesh
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = logreg.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure(1, figsize=(4, 3))
    plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)

    # Plot also the training points
    plt.scatter(X[:, 0], X[:, 1], c=Y, edgecolors='k', cmap=plt.cm.Paired)
    plt.xlabel('Sepal length')
    plt.ylabel('Sepal width')

    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xticks(())
    plt.yticks(())

    plt.show()



def IrisNaive(iris):
    X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.3,random_state=109) 
    gnb = GaussianNB()

    #Train the model using the training sets
    gnb.fit(X_train, y_train)

    #Predict the response for test dataset
    y_pred = gnb.predict(X_test)
    label.config(text="Accuracy:"+str(metrics.accuracy_score(y_test, y_pred)))
    clabel.config(text = "Confusion Matrix:"+str(metrics.confusion_matrix(y_test, y_pred)))
    # plabel.config(text = "Precision:"+str(metrics.precision_score(y_test,y_pred, average="micro")))
    rlabel.config(text=str(metrics.classification_report(y_test,y_pred)))
    

def IrisKnn(iris,k_val):
    X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2,random_state=4)
    # Label(root, 
    #      text="First Name").pack()
    # Label(root, 
    #      text="Last Name").pack()
    
    # num = int(User_input.get())
    k_r =  k_val
    k_range = range(1,k_r)
    scores = {}
    scores_list = []
    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
    label.config(text="Accuracy:"+str(metrics.accuracy_score(y_test, y_pred)))
    clabel.config(text = "Confusion Matrix:"+str(metrics.confusion_matrix(y_test, y_pred)))
    rlabel.config(text=str(metrics.classification_report(y_test, y_pred)))



# Dropdown menu options
options = [
	"Iris",
    "Breast Cancer"
]

coptions = [
    "Regression Classifier",
    "Naive Bayesian Classifier",
    "k-nn Classifier",
    "ANN Classifier",
]

# datatype of menu text
clicked = StringVar()
classifier = StringVar()

# initial menu text
# insert=Button(root, text="Upload", command=get_excel, font="none 14 bold")
clicked.set( "Select Dataset" )
classifier.set("Select Classifier")

# Create Dropdown menu
lbl = Label (root, text="Assignment No 6",  fg="black", font="none 20 bold").pack()
insert=Button(root, text="Types of Classifiers", command=show, font="none 14 bold").pack()
# insert.grid(row=26 , column=3, columnspan=2, pady=20, padx=20, ipadx=25)
drop = OptionMenu( root , clicked , *options )
drop.pack()
cdrop = OptionMenu(root,classifier, *coptions )
cdrop.pack()
# Create button, it will change label text

button = Button( root , text = "Submit" , command = show ).pack()
# cbutton = Button(root, text = "Done", command = show).pack()

# Create Label
label = Label( root , text = " " )
label.pack()
clabel = Label(root, text=" ")
clabel.pack()
plabel = Label(root, text=" ")
plabel.pack()
rlabel = Label(root, text=" ")
rlabel.pack()

# Execute tkinter
root.mainloop()
