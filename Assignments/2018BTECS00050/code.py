from tkinter import *
from tkinter import filedialog
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import os 
from scipy.stats import pearsonr
import math

def mean(dataset):
    return sum(dataset) / len(dataset)

def median(dataset):
    data = sorted(dataset)
    index = len(data) // 2

    if len(dataset) % 2 != 0:
        return data[index]

    return (data[index - 1] + data[index]) / 2


def variance(data):
    n = len(data)
    mean = sum(data) / n
    deviations = [(x - mean) ** 2 for x in data]
    variance = sum(deviations) / n
    return variance

def pearsoncoef_with_first_two(data):
	corr, _ = pearsonr(data.iloc[:,2], data.iloc[:,3])
	str=""
	str+='Pearsons correlation: %.3f' % corr
	if(corr<0):
		str+="\nNegative correlation exists"
	if(corr>0):
		str+="\nPositive correlation exists"
	if(corr==0):
		str+="\nNo correlation exists"
	print(str)

def z_score_norm(data):
	y = data.iloc[:,2]
	mean = sum(y) / len(y)
	ddof = 0
	var= sum((x - mean) ** 2 for x in y) / (len(y) - ddof)
	std_dev = math.sqrt(var)
	x_z_scaled = y
	x_z_scaled = (x_z_scaled - x_z_scaled[3]) / std_dev
	plt.scatter(x_z_scaled, x_z_scaled)
	plt.show()
	print(str(x_z_scaled))
	print("\n")

def browseFiles():
    filename = filedialog.askopenfilename(initialdir = "/", title = "Select a File", filetypes = (("all files", "*.*"), ("all files", ".")))
    label_file_explorer.configure(text="File Opened: " + filename)
    data = pd.read_csv(filename)

    print("Mean of radius: ", end =" ")
    print(round(mean(data.iloc[:,2]),5))
    print("Mean of texture: ", end =" ")
    print(round(mean(data.iloc[:,3]),5))
    print("Mean of perimeter: ", end =" ")
    print(round(mean(data.iloc[:,4]),5))
    print("Mean of area: ", end =" ")
    print(round(mean(data.iloc[:,5]),5))
    print("Mean of area: ", end =" ")
    print(round(mean(data.iloc[:,6]),5))
    print("Mean of smoothness: ", end =" ")
    print(round(mean(data.iloc[:,7]),5))
    print("Mean of symmetry: ", end =" ")
    print(round(mean(data.iloc[:,8]),5))
    print()


    print("Median of radius: ", end =" ")
    print(round(median(data.iloc[:,2]),5))
    print("Median of texture: ", end =" ")
    print(round(median(data.iloc[:,3]),5))
    print("Median of perimeter: ", end =" ")
    print(round(median(data.iloc[:,4]),5))
    print("Median of area: ", end =" ")
    print(round(median(data.iloc[:,5]),5))
    print("Median of smoothness: ", end =" ")
    print(round(median(data.iloc[:,6]),5))
    print("Median of compactness: ", end =" ")
    print(round(median(data.iloc[:,7]),5))
    print("Median of symmetry: ", end =" ")
    print(round(median(data.iloc[:,8]),5))
    print()

    print("Standard Deviation of radius: ",round(round(variance(data.iloc[:,2]),5)**0.5,5))
    print("Standard Deviation of texture: ",round(round(variance(data.iloc[:,3]),5)**0.5,5))
    print("Standard Deviation of perimeter: ",round(round(variance(data.iloc[:,4]),5)**0.5,5))
    print("Standard Deviation of area: ",round(round(variance(data.iloc[:,5]),5)**0.5,5))
    print("Standard Deviation of smoothness: ",round(round(variance(data.iloc[:,6]),5)**0.5,5))
    print("Standard Deviation of compactness: ",round(round(variance(data.iloc[:,7]),5)**0.5,5))
    print("Standard Deviation of symmetry: ",round(round(variance(data.iloc[:,8]),5)**0.5,5))
    
    print()

    # Boxplot
    data_0=np.array(data.iloc[:,2])
    data_1=np.array(data.iloc[:,3])
    data_2=np.array(data.iloc[:,4])
    data_3=np.array(data.iloc[:,5])
    data_4=np.array(data.iloc[:,6])
    data_5=np.array(data.iloc[:,7])
    data_6=np.array(data.iloc[:,8])
    
    Data=[data_0,data_1,data_2,data_3, data_4, data_5, data_6]
    plt.boxplot(Data)
    plt.title("Boxplot of all attributes")
    plt.show()
    
    # Scatter Plots
    print("Scatter Plot: Radius")
    x=np.array(data.iloc[:,2]);
    y= np.array(data.iloc[:,3]);
    plt.scatter(x, x)
    plt.title("Scatter Plot: Radius")
    plt.show()

    # Person Coefficient
    pearsoncoef_with_first_two(data)
    
    # z-score normalization
    z_score_norm(data)
    
    # x=np.array(data.iloc[:,4]);
    # y= np.array(data.iloc[:,5]);
    # plt.scatter(x, y)
    # plt.title("Petal")
    # plt.show()

    # # Q plots
    # print("Q Plot")
    # x=np.array(data[0])
    # x=sorted(x)
    # y=[]
    # k=1
    # for z in x:
    #     y.append((k-0.5)/len(x))
    #     k+=1
    # plt.scatter(y,x)
    # plt.title("Sepal Length")
    # plt.show()

    # print("Q Plot")
    # x=np.array(data[1])
    # x=sorted(x)
    # y=[]
    # k=1
    # for z in x:
    #     y.append((k-0.5)/len(x))
    #     k+=1
    # plt.scatter(y,x)
    # plt.title("Sepal Width")
    # plt.show()

    # print("Q Plot")
    # x=np.array(data[2])
    # x=sorted(x)
    # y=[]
    # k=1
    # for z in x:
    #     y.append((k-0.5)/len(x))
    #     k+=1
    # plt.scatter(y,x)
    # plt.title("Petal Length")
    # plt.show()

    # print("Q Plot")
    # x=np.array(data[3])
    # x=sorted(x)
    # y=[]
    # k=1
    # for z in x:
    #     y.append((k-0.5)/len(x))
    #     k+=1
    # plt.scatter(y,x)
    # plt.title("Petal Width")
    # plt.show()

 
    
window = Tk()

window.title('Data Analysis GUI')

window.geometry("400x200")

window.config(background = "white")

label_file_explorer = Label(window, text = "GUI to upload dataset", width = 40, height = 4, fg = "red")

button_explore = Button(window, text = "Browse Dataset", command = browseFiles)

label_file_explorer.grid(column = 0, row = 1)

button_explore.grid(column = 0, row = 2)


window.mainloop()
