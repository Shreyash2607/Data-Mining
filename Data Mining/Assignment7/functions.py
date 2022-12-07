import numpy as np
from tkinter import filedialog as fd
from tkinter import *
from csv import reader

root = Tk()
root.title("DATA MINING LAB ---> Group 05")

Console = Text(root, height = 864, width = 1536)
Console.pack()

filename = fd.askopenfilename(title = " Upload Dataset ", filetypes = (("CSV Files", "*.csv"),))

# Load a CSV file
def load_csv(filename):
	dataset = list()
	with open(filename, 'r') as file:
		csv_reader = reader(file)
		for row in csv_reader:
			if not row:
				continue
			dataset.append(row)
	return dataset

dataset = load_csv(filename)

print(type(dataset))

#print(dataset)

Y = np.array(dataset)
X = np.array([[5,3],
    [3,2],
    [4,5],
    [2,7],
    [3,3],
    [5,7],
    [7,8],
    [6,7],
    [7,5],
    [8,6],])

import matplotlib.pyplot as plt

labels = range(1, 11)
plt.figure(figsize=(10, 7))
plt.subplots_adjust(bottom=0.1)
plt.scatter(X[:,0],X[:,1], label='True Position')

for label, x, y in zip(labels, X[:, 0], X[:, 1]):
    plt.annotate(
        label,
        xy=(x, y), xytext=(-3, 3),
        textcoords='offset points', ha='right', va='bottom')
plt.show()

from scipy.cluster.hierarchy import dendrogram, linkage
from matplotlib import pyplot as plt

linked = np.array(X)

#print(type(linked))

#print(linked)

linked = linkage(X, 'single')
labelList = range(1, 11)

plt.figure(figsize=(10, 7))
dendrogram(linked,
            orientation='top',
            labels=labelList,
            distance_sort='descending',
            show_leaf_counts=True)
plt.show()

import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from sklearn.cluster import KMeans

data = pd.read_csv(filename)
data

plt.scatter(data['SepalLength'],data['SepalWidth'])
plt.xlim(-180,180)
plt.ylim(-90,90)
plt.show()

x = data.iloc[:,0:2] # 1t for rows and second for columns
x

kmeans = KMeans(3)
kmeans.fit(x)

identified_clusters = kmeans.fit_predict(x)
identified_clusters

data_with_clusters = data.copy()
data_with_clusters['Clusters'] = identified_clusters 
plt.scatter(data_with_clusters['SepalLength'],data_with_clusters['SepalWidth'],c=data_with_clusters['Clusters'],cmap='rainbow')

wcss=[]
for i in range(1,7):
    kmeans = KMeans(i)
    kmeans.fit(x)
    wcss_iter = kmeans.inertia_
    wcss.append(wcss_iter)

number_clusters = range(1,7)
plt.plot(number_clusters,wcss)
plt.title('The Elbow title')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')

