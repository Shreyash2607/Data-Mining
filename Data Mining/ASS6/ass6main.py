from tkinter import *
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from pandas.plotting import parallel_coordinates

# Import ML Libaries 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Import Dataset from sklearn 
from sklearn.datasets import load_iris

import sys
import os


# creating tkinter window
root = Tk()
root.title('Data Analysis Tool')
root.geometry("1000x1000")
NameLabel=Label(root,text="Data Analysis Tool: DM21G10",font='Arial',bg="green",padx='10px',pady='10px',fg="white").pack()
NameLabel2=Label(root,text="ASS6: Classifiers",font='Arial',bg="azure",padx='8px',pady='5px',fg="black").pack()

def selected(event):
    myLabel=Label(root,text="Dataset -> "+clicked.get()+" selected",bg="white",padx='4px',pady='4px').place(x=50,y=130)
    

options=[
    "IRIS"
    #"Breast-cancer data set"
]
clicked = StringVar()
clicked.set(" Select Dataset")
drop = OptionMenu(root,clicked,*options,command=selected)
drop.place(x=50,y=90)

def attrselected(event):
    attrLabel=Label(root,text="Classifier-> "+attrclicked.get()+" selected",bg="white",padx=3,pady=3).place(x=50,y=210)

    if clicked.get()=="IRIS":
        if attrclicked.get()=="Regression classifier":
            os.system('plot_iris_logistic.py')
        if attrclicked.get()=="Naïve Bayesian Classifier":
            os.system('iris_Naive_Bayes.py')
        if attrclicked.get()=="k-NN classifier":
            os.system('iris_kNN.py')
        if attrclicked.get()=="Three layer Artificial Neural Network (ANN) classifier":
            os.system('iris_3ANNback.py')

    if clicked.get()=="Breast-cancer data set":
        data="C:/Users/revati/Desktop/Academics/sem 7th/DM/ASS4/balance_scale_dataset/balnce_scale_data_set.csv"
        #call the function
        #breast_cancer_DT(data)
        
        


attroptions=[
    "Regression classifier",
    "Naïve Bayesian Classifier",
    "k-NN classifier",
    "Three layer Artificial Neural Network (ANN) classifier"    
]

attrclicked = StringVar()
attrclicked.set("Select Classifier")
attrdrop = OptionMenu(root,attrclicked,*attroptions,command=attrselected)
attrdrop.place(x=50,y=170)

root.mainloop()