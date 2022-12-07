import pandas as pd
import csv

data = pd.read_csv('car_evaluation.csv',usecols=['buying price','maintainence cost','number of doors','number of persons','lug_boot','safety','decision'])
class1 = data['decision'].value_counts()['unacc']
class2 = data['decision'].value_counts()['acc']



data1 = list(csv.reader(open('car_evaluation.csv')))
count = 0
a = False
b=False
for row in data1:
    for col in row:
        if col == "vhigh":
            a=True
        if col == "unacc":
            b=True
    if a and b:
        count+=1
    a=False
    b=False

print(count)
print(data1[1])