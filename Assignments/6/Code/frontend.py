from tkinter import *
from tkinter.filedialog import askopenfilename
from tkinter import messagebox as mb

from numpy import complex128
import backend
window = Tk()

window.geometry('750x300')
window.title("Data Analysis Tool")
# window.resizable(0,0)

############################################################################################################
# 1. Global Variables to be updated dynamically from anywhere
## 1.1 Variable to hold details of the file uploaded
fileDetails = StringVar()
fileDetails.set("File not uploaded")

## 1.2 Recognition rate value for selected classifier
recognitionRate = StringVar()
recognitionRate.set("Recognition Rate: NA")

## 1.3 Misclassification Rate for selected classifier
misclassificationRate = StringVar()
misclassificationRate.set("Misclassification Rate: NA")

## 1.4 Sensitivity for selected classifier
sensitivity = StringVar()
sensitivity.set("Sensitivity: NA")

## 1.5 Specificity for selected classifier
specificity = StringVar()
specificity.set("Specificity: NA")

## 1.6 precisionAndRecall for selected classifier
precisionAndRecall = StringVar()
precisionAndRecall.set("Precision And Recall : NA")

############################################################################################################
# 2. The Navbar
## Function to open new file and change file details
def OpenFile():
    filepath = askopenfilename()
    filename = filepath.split("/")
    filename.reverse()
    print(filepath)
    fileDetails.set("New file uploaded.\nFile Name : " + filename[0] + "\nAttributes : ")
## Function to show product info
def About():
    mb.showinfo("Developers Info", "Designed and built by :\n2018BTECS00050 Rushikesh Ramesh Shelke\n2018BTECS00064 Saurabh Raghunath Hirugade")

def HowToUse():
    mb.showinfo("Product Info", "1.First upload a .csv/.excel file of sample data \n2.Click on any button to perform data analysis\n")

menu = Menu(window)
window.config(menu=menu)

## Nav item: File
filemenu = Menu(menu)
menu.add_cascade(label="File", menu=filemenu)
filemenu.add_command(label="Upload", command=OpenFile)
filemenu.add_separator()
filemenu.add_command(label="Exit", command=window.quit)


## Nav item: Help
helpmenu = Menu(menu)
menu.add_cascade(label="Help", menu=helpmenu)
helpmenu.add_command(label="How to use", command=HowToUse)
helpmenu.add_separator()
helpmenu.add_command(label="About", command=About)

############################################################################################################
# 3. File Info Box
topframe = Frame(window, bg="#faf8e9")
topframe.pack(side=TOP)

## Frame Title:
TopFrameTitle = Label(topframe, text="File Details :", bg="#faf8e9")
# TopFrameTitle.pack(side=LEFT)
TopFrameTitle.grid(row=0, column=0)

## Frame Details: 
# FileInfoTextBox = Text(topframe, height=2, width=30, bg="White")
# FileInfoTextBox.grid(column=0, row=0)
# FileInfoTextBox.pack(fill=X, side=LEFT)
# FileInfoTextBox.insert(END, )
FileInfoTextBox = Label(topframe, text="File not uploaded", textvariable=fileDetails, bg="white", font=('Verdana', 10, 'italic'))
# FileInfoTextBox.pack(side=LEFT)
FileInfoTextBox.grid(row=0, column=1)



############################################################################################################
# 4. Options list for data analysis

## 4.1 Decision Tree classifier
leftframe = Frame(window, bg="#0f163b", padx=5, pady=5, borderwidth=5, relief=RAISED)
leftframe.pack(side=LEFT)

rowVal = 0
rowVal+=1
DTC = Label(leftframe, text="Select Classifier :-", fg="white", bg="#0f163b", font=('Arial', 12, 'bold', 'italic'))
DTC.grid(column=0, row=rowVal, columnspan=3, pady=1)

rowVal+=1
IG = Button(leftframe, text="Regression Classifier")
IG.grid(column=0, row=rowVal, padx=2, pady=2)
rowVal+=1
GR = Button(leftframe, text="Naive Bayes Classifier")
GR.grid(column=0, row=rowVal, padx=2, pady=2)
rowVal+=1
GI = Button(leftframe, text="k-NN classifier")
GI.grid(column=0, row=rowVal, padx=2, pady=2)
rowVal+=1
ANN = Button(leftframe, text="Three layer ANN")
ANN.grid(column=0, row=rowVal, padx=2, pady=2)


## 4.2 Performance
rightframe = Frame(window, bg="#0f163b", padx=5, pady=5, borderwidth=5, relief=RAISED)
rightframe.pack(side=RIGHT)

rowVal+=1
Performance = Label(rightframe, text="Performance Details :", fg="white", bg="#0f163b", font=('Arial', 12, 'bold', 'italic'))
Performance.grid(column=0, row=rowVal, pady=1, columnspan=3)

### 4.2.1 Recognition rate
rowVal+=1
RecognitionRate = Label(rightframe, text="", textvariable=recognitionRate, fg="white", bg="#0f163b")
RecognitionRate.grid(column=0, row=rowVal, columnspan=3, sticky=W)
### 4.2.2 Misclassification rate
rowVal+=1
MisclassificationRate = Label(rightframe, text="", textvariable=misclassificationRate, fg="white", bg="#0f163b")
MisclassificationRate.grid(column=0, row=rowVal, columnspan=3, sticky=W)
### 4.2.3 Sensitivity
rowVal+=1
Sensitivity = Label(rightframe, text="", textvariable=sensitivity, fg="white", bg="#0f163b")
Sensitivity.grid(column=0, row=rowVal, columnspan=3, sticky=W)
### 4.2.4 Specificity
rowVal+=1
Specificity = Label(rightframe, text="", textvariable=specificity, fg="white", bg="#0f163b")
Specificity.grid(column=0, row=rowVal, columnspan=3, sticky=W)
### 4.2.5 Precision and recall
rowVal+=1
PrecisionAndRecall = Label(rightframe, text="", textvariable= precisionAndRecall, fg="white", bg="#0f163b")
PrecisionAndRecall.grid(column=0, row=rowVal, columnspan=3, sticky=W)

############################################################################################################
# 5.Display of confusion matrix

TP = IntVar()
TP.set(0)

FP = IntVar()
FP.set(0)

TN = IntVar()
TN.set(0)

FN = IntVar()
FN.set(0)

centerframe = Frame(window, bg="#0f163b", padx=5, pady=5, borderwidth=5, relief=RAISED)
centerframe.pack(side=BOTTOM)

C00 = Label(centerframe, text="Perdicted ➡\nActual ⬇", bd=5)
C00.grid(row=0, column=0)
C01 = Label(centerframe, text="Positve")
C01.grid(row=0, column=1)
C02 = Label(centerframe, text="Negative")
C02.grid(row=0, column=2)

C10 = Label(centerframe, text="Positve")
C10.grid(row=1, column=0)
C11 = Label(centerframe, text="", textvariable=TP)
C11.grid(row=1, column=1)
C12 = Label(centerframe, text="", textvariable=FP)
C12.grid(row=1, column=2)

C20 = Label(centerframe, text="Negative")
C20.grid(row=2, column=0)
C21 = Label(centerframe, text="", textvariable=FN)
C21.grid(row=2, column=1)
C22 = Label(centerframe, text="", textvariable=TN)
C22.grid(row=2, column=2)
mainloop()