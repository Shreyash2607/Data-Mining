from csv import reader
from collections import defaultdict
from itertools import chain, combinations
from optparse import OptionParser
from tkinter import *
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from apyori import apriori

def dataToCSV(fname):
    first = True
    currentID = 1
    with open(fname, 'r') as dataFile, open(fname + '.csv', 'w') as outputCSV:
        for line in dataFile:
            nums = line.split()
            itemSetID = nums[1]
            item = nums[2]
            if(int(itemSetID) == currentID):
                if(first):
                    outputCSV.write(item)
                else:
                    outputCSV.write(',' + item)
                first = False
            else:
                outputCSV.write('\n' + item)
                currentID += 1


def powerset(s):
    return chain.from_iterable(combinations(s, r) for r in range(1, len(s)))


def getFromFile(fname):
    itemSets = []
    itemSet = set()

    with open(fname, 'r') as file:
        csv_reader = reader(file)
        for line in csv_reader:
            line = list(filter(None, line))
            record = set(line)
            for item in record:
                itemSet.add(frozenset([item]))
            itemSets.append(record)
    return itemSet, itemSets


def getAboveMinSup(itemSet, itemSetList, minSup, globalItemSetWithSup):
    freqItemSet = set()
    localItemSetWithSup = defaultdict(int)

    for item in itemSet:
        for itemSet in itemSetList:
            if item.issubset(itemSet):
                globalItemSetWithSup[item] += 1
                localItemSetWithSup[item] += 1

    for item, supCount in localItemSetWithSup.items():
        support = float(supCount / len(itemSetList))
        if(support >= minSup):
            freqItemSet.add(item)

    return freqItemSet


def getUnion(itemSet, length):
    return set([i.union(j) for i in itemSet for j in itemSet if len(i.union(j)) == length])


def pruning(candidateSet, prevFreqSet, length):
    tempCandidateSet = candidateSet.copy()
    for item in candidateSet:
        subsets = combinations(item, length)
        for subset in subsets:
            if(frozenset(subset) not in prevFreqSet):
                tempCandidateSet.remove(item)
                break
    return tempCandidateSet



def associationRule(freqItemSet, itemSetWithSup, minConf):
    rules = []
    for k, itemSet in freqItemSet.items():
        for item in itemSet:
            subsets = powerset(item)
            for s in subsets:
                confidence = float(
                    itemSetWithSup[item] / itemSetWithSup[frozenset(s)])
                if(confidence > minConf):
                    rules.append([set(s), set(item.difference(s)), confidence])
    return rules



def getItemSetFromList(itemSetList):
    tempItemSet = set()

    for itemSet in itemSetList:
        for item in itemSet:
            tempItemSet.add(frozenset([item]))

    return tempItemSet

# def apriori1(itemSetList, minSup, minConf):
#     C1ItemSet = getItemSetFromList(itemSetList)
#     globalFreqItemSet = dict()
#     globalItemSetWithSup = defaultdict(int)

#     L1ItemSet = getAboveMinSup(
#         C1ItemSet, itemSetList, minSup, globalItemSetWithSup)
#     currentLSet = L1ItemSet
#     k = 2

#     while(currentLSet):
#         globalFreqItemSet[k-1] = currentLSet
#         candidateSet = getUnion(currentLSet, k)
#         candidateSet = pruning(candidateSet, currentLSet, k-1)
#         currentLSet = getAboveMinSup(
#             candidateSet, itemSetList, minSup, globalItemSetWithSup)
#         k += 1

#     rules = associationRule(globalFreqItemSet, globalItemSetWithSup, minConf)
#     rules.sort(key=lambda x: x[2])

#     return globalFreqItemSet, rules

def aprioriFromFile(fname, minSup, minConf):
    C1ItemSet, itemSetList = getFromFile(fname)

    globalFreqItemSet = dict()
    globalItemSetWithSup = defaultdict(int)

    L1ItemSet = getAboveMinSup(
        C1ItemSet, itemSetList, minSup, globalItemSetWithSup)
    currentLSet = L1ItemSet
    k = 2

    while(currentLSet):
        globalFreqItemSet[k-1] = currentLSet
        candidateSet = getUnion(currentLSet, k)
        candidateSet = pruning(candidateSet, currentLSet, k-1)
        currentLSet = getAboveMinSup(
            candidateSet, itemSetList, minSup, globalItemSetWithSup)
        k += 1

    rules = associationRule(globalFreqItemSet, globalItemSetWithSup, minConf)
    rules.sort(key=lambda x: x[2])

    return globalFreqItemSet, rules



# def write(*message, end = "\n", sep = " "):
#     text = ""
#     for item in message:
#         text += "{}".format(item)
#         text += sep
#     text += end
#     text += "\n"
#     Console.insert(INSERT, text)


def calculate ():  
    x1 = entry1.get()
    x2 = entry2.get()
    
    root = Tk()
    root.title("DM21G16")

    Console = Text(root, height = 864, width = 1535)
    Console.pack()

    freqItemSet, rules = aprioriFromFile("house-votes-84.data", float(x1), float(x2))
    #write(freqItemSet)



    text = "Frequency Item Set  -> \n\n"
    for item in freqItemSet:
        text += "{}".format(item)
        text += " "
    text += "\n"
    text += "\n"
    Console.insert(INSERT, text)

    text = "\n\nRules ===> \n\n"
    for item in rules:
        text += "{}".format(item)
        text += "\n"
    text += "\n"
    text += "\n"
    Console.insert(INSERT, text)
    #write(rules)

    store_data = pd.read_csv('house-votes-84.csv', header=None)

    store_data.head()

    records = []
    for i in range(0, 430):
        records.append([str(store_data.values[i,j]) for j in range(0, 15)])

    association_rules = apriori(records, min_length = 10, min_support = 0.000001*float(x1), min_confidence  = float(x2), min_lift  = 0.5)
    association_results = list(association_rules)

    text = "\n\n Assotiation Rules Length = " + str(len(association_results)) + "\n\n"

    for item in association_results:
        text += "{}".format(item)
        text += "\n"
    text += "\n"
    text += "\n\n"
    Console.insert(INSERT, text)

    text = "\n\n Rules ===> \n\n"

    for item in association_results:
        pair = item[0] 
        items = [x for x in pair]
        # text += ("Rule: " + items[0] + " -> " + items[1] + "\n")

        text += ("Support: " + str(item[1]) + "\n")

        text += (("Confidence: " + str(item[2][0][2])) + "\n")
        # text += (("Lift: " + str(item[2][0][3])) + "\n")
        text += "=====================================\n"
        
    Console.insert(INSERT, text)

root= Tk()

canvas1 = Canvas(root, width = 400, height = 300)
canvas1.pack()

support_label = Label(root, text = "Support").place(x = 80, y = 130)  

entry1 = Entry (root) 
canvas1.create_window(200, 140, window=entry1)

confidence_label = Label(root, text = "Confidence").place(x = 60, y = 150)

entry2 = Entry (root)
canvas1.create_window(200, 160, window=entry2)

button1 = Button(text='Calculate', command=calculate)
canvas1.create_window(200, 180, window=button1)


# def write(*message, end = "\n", sep = " "):
#     text = ""
#     for item in message:
#         text += "{}".format(item)
#         text += sep
#     text += end
#     Console.insert(INSERT, text)

root.mainloop()

