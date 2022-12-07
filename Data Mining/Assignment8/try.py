# import os
# os.environ["MKL_NUM_THREADS"] = "1" 
# os.environ["NUMEXPR_NUM_THREADS"] = "1" 
# os.environ["OMP_NUM_THREADS"] = "1" 

# import numpy as np
# import pandas as pd

# class DensePageRank:
#     def load_graph_dataset(self, data_home, is_undirected=False):
#         '''
#         Load the graph dataset from the given directory (data_home)

#         inputs:
#             data_home: string
#                 directory path conatining a dataset
#             is_undirected: bool
#                 if the graph is undirected
#         '''
#         # Step 1. set file paths from data_home
#         edge_path = "{}/edges.tsv".format(data_home)

#         # Step 2. read the list of edges from edge_path
#         edges = np.loadtxt(edge_path, dtype=int)
#         n = int(np.amax(edges[:, 0:2])) + 1 # the current n is the maximum node id (starting from 0)

#         # Step 3. convert the edge list to the adjacency matrix
#         self.A = np.zeros((n, n))
#         for i in range(edges.shape[0]):
#             source, target, weight = edges[i, :]
#             self.A[(source, target)] = weight
#             if is_undirected:
#                 self.A[(target, source)] = weight

#         # Step 4. set n (# of nodes) and m (# of edges)
#         self.n = n                         # number of nodes
#         self.m = np.count_nonzero(self.A)  # number of edges

# class DensePageRank(DensePageRank):
#     def load_node_labels(self, data_home):
#         '''
#         Load the node labels from the given directory (data_home)

#         inputs:
#             data_home: string
#                 directory path conatining a dataset
#         '''
#         label_path = "{}/node_labels.tsv".format(data_home)
#         self.node_labels = pd.read_csv(label_path, sep="\t")

# data_home = './data/small'
# dpr = DensePageRank()
# dpr.load_graph_dataset(data_home, is_undirected=True)
# dpr.load_node_labels(data_home)

# # print the number of nodes and edges
# print("The number n of nodes: {}".format(dpr.n))
# print("The number m of edges: {}".format(dpr.m))

# # print the heads (5) of the node labels
# display(dpr.node_labels.head(5))

import numpy as nm  
import matplotlib.pyplot as mtp  
import pandas as pd  


#Importing the dataset  
dataset = pd.read_csv('Market_Basket_Optimisation.csv')  
transactions=[]  
for i in range(0, 100):  
    transactions.append([str(dataset.values[i,j])  for j in range(0,20)])  

from apyori import apriori
rules= apriori(transactions= transactions, min_support=0.003, min_confidence = 0.2, min_lift=3, min_length=2, max_length=2)

results= list(rules)  
results   

for item in results:  
    pair = item[0]   
    items = [x for x in pair]  
    print("Rule: " + items[0] + " -> " + items[1])  
  
    print("Support: " + str(item[1]))  
    print("Confidence: " + str(item[2][0][2]))  
    print("Lift: " + str(item[2][0][3]))  
    print("=====================================")  