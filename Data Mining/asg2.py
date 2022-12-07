import tkinter as tk
from tkinter import ttk
from tkinter import *
from tkinter import messagebox
from tkinter import filedialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import pylab as py
import statsmodels.api as sm
import matplotlib.pyplot as plt
import pandas as pd
import math
import scipy.stats as stats
import pylab 
import plotly.express as px

root= tk.Tk()
root.title("Assignment No. 2 ")
root.geometry("1000x1000")
root.configure(bg='palegreen')
lbl = Label (root, text="   Assignment No 2:DM21G05", justify='center', fg="red", font="none 20 bold")
lbl.grid(column=0, row=0, sticky=tk.N, padx=10, pady=10)

y=[13,15,16,16,19,20,20,21,22,22,25,25,25,25,30,33,33,35,35,35,35,36,40,46,52,70]

def get_excel():
	import_file_path = filedialog.askopenfilename()

def findMean():
	n = len(y) 
	get_sum = sum(y) 
	mean = get_sum / n 
	ans="Mean is: " + str(mean)
	print(ans)

def findMedian():
	n = len(y) 
	newy=sorted(y)
	if n % 2 == 0: 
	    median1 = newy[n//2] 
	    median2 = newy[n//2 - 1] 
	    median = (median1 + median2)/2
	else: 
	    median = newy[n//2] 
	ans="Median is: " + str(median)
	print(ans)

def findMode():
	from collections import Counter
  
	n = len(y) 
	  
	data = Counter(y) 
	get_mode = dict(data) 
	mode = [k for k, v in get_mode.items() if v == max(list(data.values()))] 
	  
	if len(mode) == n: 
	    get_mode = "No mode found"
	else: 
	    get_mode = "Mode is / are: " + ', '.join(map(str, mode)) 
	      
	#root.t3.insert(END, str(get_mode))
	print(get_mode)

def findMidrange():
	n=len(y)
	data=sorted(y) 
	maxV=data[n-1]
	minV=data[0]
	ans_midrange=(minV+maxV)/2
	ans="Midrange: "+str(ans_midrange)
	print(ans)

def findvariance():
	n = len(y)
	sum_y=sum(y)
	mean =sum_y/n
	deviations = [(x - mean) ** 2 for x in y]
	variance = sum(deviations) / n
	ans="Variance: "+str(variance)
	print(ans)

def findstddeviation():
	ddof=0
	n = len(y)
	mean = sum(y) / n
	var= sum((x - mean) ** 2 for x in y) / (n - ddof)
	std_dev = math.sqrt(var)
	ans="Standard Deviation: "+str(std_dev)
	print(ans)

def findrange():
	n=len(y)
	data=sorted(y) 
	maxV=data[n-1]
	minV=data[0]
	ans_range=maxV-minV
	ans="Range: "+str(ans_range)
	print(ans)

def calc_quantile():
    s_lst = sorted(y)
    ans=[]
    res="Quantiles are: "
    i=0.25
    for j in range(0,4):
	    idx = (len(s_lst) - 1)*i
	    int_idx = int(idx)
	    remainder = idx % 1
	    if remainder > 0:
	        lower_val = s_lst[int_idx]
	        upper_val = s_lst[int_idx + 1]
	        ans.append(lower_val * (1 - remainder) + upper_val * remainder)
	    else:
	    	ans.append(s_lst[int_idx])
	    i=i+0.25
	    res=str(j+1)+"th Quantile: "+str(ans[j])
	    print(res)

def calc_quantile_range():
	s_lst = sorted(y)
	ans=[]
	i=0.25
	for j in range(0,4):
	    idx = (len(s_lst) - 1)*i
	    int_idx = int(idx)
	    remainder = idx % 1
	    if remainder > 0:
	        lower_val = s_lst[int_idx]
	        upper_val = s_lst[int_idx + 1]
	        ans.append(lower_val * (1 - remainder) + upper_val * remainder)
	    else:
	    	ans.append(s_lst[int_idx])
	    i=i+0.25
	ans="Interquantile Range: "+str(ans[2]-ans[0])
	print(ans)

def findfive_number_summary():
	n = len(y) 
	newy=sorted(y)
	if n % 2 == 0: 
	    median1 = newy[n//2] 
	    median2 = newy[n//2 - 1] 
	    median = (median1 + median2)/2
	else: 
	    median = newy[n//2] 
	med_ans="Median is: " + str(median)
	s_lst = sorted(y)
	sum_ans=[]
	i=0.25
	for j in range(0,4):
	    idx = (len(s_lst) - 1)*i
	    int_idx = int(idx)
	    remainder = idx % 1
	    if remainder > 0:
	        lower_val = s_lst[int_idx]
	        upper_val = s_lst[int_idx + 1]
	        sum_ans.append(lower_val * (1 - remainder) + upper_val * remainder)
	    else:
	    	sum_ans.append(s_lst[int_idx])
	    i=i+0.25
	res=med_ans+"\n1st Quartile: "+str(sum_ans[0])+"\n3rd Quartile: "+str(sum_ans[2])+"\nMinimum Element: "+str(min(y))+"\nMaximum Element: "+str(max(y))
	print("Five Number Summary:")
	print(res)

def qq_plot():
	s_lst = sorted(y)
	ans=[]
	res="Quantiles are: "
	i=0.25
	for j in range(0,4):
	    idx = (len(s_lst) - 1)*i
	    int_idx = int(idx)
	    remainder = idx % 1
	    if remainder > 0:
	        lower_val = s_lst[int_idx]
	        upper_val = s_lst[int_idx + 1]
	        ans.append(lower_val * (1 - remainder) + upper_val * remainder)
	    else:
	    	ans.append(s_lst[int_idx])
	    i=i+0.25
	stats.probplot(ans, dist="norm", plot=pylab)
	pylab.show()

def histogram():
	plt.hist(y)
	plt.show()

def scatter_plot():
	plt.scatter(y, y)
	plt.show()

def box_plot():
	plt.boxplot(y)
	plt.show()

insert=Button(root, text="Upload", command=get_excel, font="none 14 bold")
insert.grid(row=2, column=0, columnspan=2, pady=30, padx=30, ipadx=50)

insert=Button(root, text="Mean", command=findMean, font="none 14 bold")
insert.grid(row=5, column=0, columnspan=2, pady=20, padx=20, ipadx=25)

insert=Button(root, text="Median", command=findMedian, font="none 14 bold")
insert.grid(row=5, column=3, columnspan=2, pady=20, padx=20, ipadx=25)

insert=Button(root, text="Mode", command=findMode, font="none 14 bold")
insert.grid(row=5, column=6, columnspan=2, pady=20, padx=20, ipadx=25)

insert=Button(root, text="Std. Deviation", command=findstddeviation, font="none 14 bold")
insert.grid(row=8, column=0, columnspan=2, pady=20, padx=20, ipadx=25)

insert=Button(root, text="Midrange", command=findMidrange, font="none 14 bold")
insert.grid(row=8, column=3, columnspan=2, pady=20, padx=20, ipadx=25)

insert=Button(root, text="Variance", command=findvariance, font="none 14 bold")
insert.grid(row=8, column=6, columnspan=2, pady=20, padx=20, ipadx=25)

insert=Button(root, text="Range", command=findrange, font="none 14 bold")
insert.grid(row=11, column=0, columnspan=2, pady=30, padx=30, ipadx=25)

insert=Button(root, text="Quartile", command=calc_quantile, font="none 14 bold")
insert.grid(row=11, column=3, columnspan=2, pady=30, padx=30, ipadx=25)

insert=Button(root, text="Interquartile range ", command=calc_quantile_range, font="none 14 bold")
insert.grid(row=11, column=6, columnspan=2, pady=30, padx=30, ipadx=50)

insert=Button(root, text="Five-number summary", command=findfive_number_summary, font="none 14 bold")
insert.grid(row=14, column=0, columnspan=2, pady=30, padx=30, ipadx=50)

insert=Button(root, text="Quantile Plot", command=qq_plot, font="none 14 bold")
insert.grid(row=14, column=3, columnspan=2, pady=20, padx=20, ipadx=25)

insert=Button(root, text="Q-Q Plot", command=qq_plot, font="none 14 bold")
insert.grid(row=14, column=6, columnspan=2, pady=20, padx=20, ipadx=25)

insert=Button(root, text="Histogram", command=histogram, font="none 14 bold")
insert.grid(row=17, column=0, columnspan=2, pady=20, padx=20, ipadx=25)

insert=Button(root, text="Scatter Plot", command=scatter_plot, font="none 14 bold")
insert.grid(row=17, column=3, columnspan=2, pady=20, padx=20, ipadx=25)

insert=Button(root, text="Box Plot", command=box_plot, font="none 14 bold")
insert.grid(row=17, column=6, columnspan=2, pady=20, padx=20, ipadx=25)

root.mainloop()