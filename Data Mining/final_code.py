import tkinter as tk
from tkinter import ttk
from tkinter import *
from tkinter import filedialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import pylab as py
import pylab
import math
from tkinter import messagebox
import statsmodels.api as sm
import plotly.express as px
from scipy.stats import chi2_contingency
from scipy.stats import chi2
from numpy.random import randn
from numpy.random import seed
from scipy.stats import pearsonr
import numpy as np

root= tk.Tk()

canvas1 = tk.Canvas(root, width = 1300, height = 500)
canvas1.pack()
 
label1 = tk.Label(root, text='Data Analysis Tool: DM21G10')
label1.config(font=('Arial', 20))
canvas1.create_window(510, 50, window=label1)
 
def getExcel ():
      global df
 
      import_file_path = filedialog.askopenfilename()
      df = pd.read_excel(import_file_path)
      global y
      global x
      #x = df['Name']
      y = df['A']
      x = df['B']
      
      clear_treeview()

      tree["column"] = list(df.columns)
      tree["show"] = "headings"

      for col in tree["column"]:
      	tree.heading(col, text=col)

      df_rows = df.to_numpy().tolist()
      for row in df_rows:
      		tree.insert("", "end", values=row)

      tree.pack()

# Clear the Treeview Widget
def clear_treeview():
   tree.delete(*tree.get_children())

frame=Frame(root)
frame.pack(pady=20)
# Create a Treeview widget
tree = ttk.Treeview(frame)


# Add a Label widget to display the file content
label = Label(root, text='')
label.pack(pady=20)
def clear_charts():
      bar1.get_tk_widget().pack_forget()

def findMean():
	n = len(y) 
	get_sum = sum(y) 
	mean = get_sum / n 
	ans="Mean is: " + str(mean)
	root.t1.insert(END, str(ans))

def findMedian():
	#df.Marks = pd.to_numeric(df.Marks, errors='coerce')
	#df.sort_values(['Marks'],inplace=True) 
	#y=df['Marks']
	n = len(y) 
	newy=sorted(y)
	if n % 2 == 0: 
	    median1 = newy[n//2] 
	    median2 = newy[n//2 - 1] 
	    median = (median1 + median2)/2
	else: 
	    median = newy[n//2] 
	ans="Median is: " + str(median)
	root.t2.insert(END, str(ans))

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
	      
	root.t3.insert(END, str(get_mode))

def findMidrange():
	n=len(y)
	data=sorted(y) 
	maxV=data[n-1]
	minV=data[0]
	ans_midrange=(minV+maxV)/2
	root.t4.insert(END, str(ans_midrange))

def findvariance():
	n = len(y)
	sum_y=sum(y)
	mean =sum_y/n
	deviations = [(x - mean) ** 2 for x in y]
	variance = sum(deviations) / n
	root.t5.insert(END, str(variance))

def findstddeviation():
	ddof=0
	n = len(y)
	mean = sum(y) / n
	var= sum((x - mean) ** 2 for x in y) / (n - ddof)
	std_dev = math.sqrt(var)
	root.t6.insert(END, str(std_dev))

def findrange():
	n=len(y)
	data=sorted(y) 
	maxV=data[n-1]
	minV=data[0]
	ans_range=maxV-minV
	root.t7.insert(END, str(ans_range))

def findquartiles():
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
	        #messagebox.showinfo("Quartile", lower_val * (1 - remainder) + upper_val * remainder)
	    else:
	    	ans.append(s_lst[int_idx])
	    	#messagebox.showinfo("Quartile",s_lst[int_idx])
	    i=i+0.25
	    res=str(j+1)+"th Quantile: "
	    #messagebox.showinfo("Quantiles",res+str(ans[j]))
	root.t8.insert(END, str(ans))

def findinterquartilerange():
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
	        #messagebox.showinfo("Quartile", lower_val * (1 - remainder) + upper_val * remainder)
	    else:
	    	ans.append(s_lst[int_idx])
	    	#messagebox.showinfo("Quartile",s_lst[int_idx])
	    i=i+0.25
	#messagebox.showinfo("Interquartile Range:",ans[2]-ans[0])
	root.t9.insert(END, str(ans[2]-ans[0]))

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
	        #messagebox.showinfo("Quartile", lower_val * (1 - remainder) + upper_val * remainder)
	    else:
	    	sum_ans.append(s_lst[int_idx])
	    	#messagebox.showinfo("Quartile",s_lst[int_idx])
	    i=i+0.25
	res=med_ans+"\n 1st Quartile: "+str(sum_ans[0])+"\n 3rd Quartile: "+str(sum_ans[2])+"\n Minimum Element: "+str(min(y))+"\nMaximum Element: "+str(max(y))
	messagebox.showinfo("Five Numbers Summary",res)
	
def histo():
	plt.hist(y)
	plt.show()

def qq():
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
	        #messagebox.showinfo("Quartile", lower_val * (1 - remainder) + upper_val * remainder)
	    else:
	    	ans.append(s_lst[int_idx])
	    	#messagebox.showinfo("Quartile",s_lst[int_idx])
	    i=i+0.25
	stats.probplot(ans, dist="norm", plot=pylab)
	pylab.show()

def scat():
	plt.scatter(y, y)
	plt.show()

def box_plot():
	plt.boxplot(y)
	plt.show()

def chi2test_with_x_y():
		table = [x,y]
		stat, p, dof, expected = chi2_contingency(table)
		prob = 0.95
		critical = chi2.ppf(prob, dof)
		if abs(stat) >= critical:
			print("Dependent (reject H0)")
		else:
			print("Independent (fail to reject H0)")
		print(str(expected))

def pearsoncoef_with_x_y():
		corr, _ = pearsonr(x, y)
		str=""
		str+='Pearsons correlation: %.3f' % corr
		if(corr<0):
			str+="\nNegative correlation exists"
		if(corr>0):
			str+="\nPositive correlation exists"
		if(corr==0):
			str+="\nNo correlation exists"
		print(str)

def covariance_with_x_y():
		cov_mat = np.stack((x, y), axis = 0)
		print(str(cov_mat))
		cov_mat = np.stack((x, y), axis = 1)
		print(str(cov_mat))

def min_max_norm_x():
		x_min_max_scaled = x.copy()
		x_min_max_scaled = (x_min_max_scaled - x_min_max_scaled.min()) / (x_min_max_scaled.max() - x_min_max_scaled.min())
		plt.scatter(x_min_max_scaled, x_min_max_scaled)
		plt.show()
		print(str(x_min_max_scaled))

def z_score_norm_x():
		x_z_scaled = x.copy()
		x_z_scaled = (x_z_scaled - x_z_scaled.mean()) / x_z_scaled.std()
		plt.scatter(x_z_scaled, x_z_scaled)
		plt.show()
		print(str(x_z_scaled))

def dec_scale_norm_x():
		p = x.max()
		q = len(str(abs(p)))
		x_des_scaled = x/10**q
		plt.scatter(x_des_scaled, x_des_scaled)
		plt.show()
		print(str(x_des_scaled))

browseButton_Excel = tk.Button(text='Load File...', command=getExcel, bg='green', fg='white', font=('helvetica', 12, 'bold'))
canvas1.create_window(430, 100, window=browseButton_Excel)

buttonExit = tk.Button (root, text='Exit!', command=root.destroy,fg='white', bg='red', font=('helvetica', 11, 'bold'))
canvas1.create_window(600, 100, window=buttonExit)

Central=tk.Button (root, text='Central tendency', bg='white',fg='black', font=('helvetica', 11, 'bold'))
canvas1.create_window(200, 190, window=Central)

button2a = tk.Button (root, text='Mean', command=findMean, bg='black',fg='white', font=('helvetica', 11, 'bold'))
canvas1.create_window(100, 220, window=button2a)

button2b = tk.Button (root, text='Median', command=findMedian, bg='black',fg='white', font=('helvetica', 11, 'bold'))
canvas1.create_window(100, 260, window=button2b)

button2c = tk.Button (root, text='Mode', command=findMode, bg='black',fg='white', font=('helvetica', 11, 'bold'))
canvas1.create_window(100, 300, window=button2c)

button2d = tk.Button (root, text='Midrange', command=findMidrange, bg='black',fg='white', font=('helvetica', 11, 'bold'))
canvas1.create_window(100, 340, window=button2d)

button2e = tk.Button (root, text='Variance', command=findvariance, bg='black',fg='white', font=('helvetica', 11, 'bold'))
canvas1.create_window(100, 380, window=button2e)

button2f = tk.Button (root, text='Standard Deviation', command=findstddeviation, bg='black',fg='white', font=('helvetica', 11, 'bold'))
canvas1.create_window(100, 420, window=button2f)

#canvas1.create_line(400,180,400,480)
dispersion=tk.Button (root, text='Dispersion of data', bg='white',fg='black', font=('helvetica', 11, 'bold'))
canvas1.create_window(510, 190, window=dispersion)

rangebut = tk.Button (root, text='Range', command=findrange, bg='black',fg='white', font=('helvetica', 11, 'bold'))
canvas1.create_window(400, 230, window=rangebut)

quartiles_but = tk.Button (root, text='Quartiles', command=findquartiles, bg='black',fg='white', font=('helvetica', 11, 'bold'))
canvas1.create_window(400, 270, window=quartiles_but)

intquartile_range_but = tk.Button (root, text='IQR', command=findinterquartilerange, bg='black',fg='white', font=('helvetica', 11, 'bold'))
canvas1.create_window(400, 310, window=intquartile_range_but)

five_number_summary_but=tk.Button (root, text='FiveNumbersummary', command=findfive_number_summary, bg='black',fg='white', font=('helvetica', 11, 'bold'))
canvas1.create_window(450, 360, window=five_number_summary_but)

graphical=tk.Button (root, text='Graphical Display', bg='white',fg='black', font=('helvetica', 11, 'bold'))
canvas1.create_window(700, 190, window=graphical)

histo=tk.Button (root, text='Histogram', command=histo, bg='black',fg='white', font=('helvetica', 11, 'bold'))
canvas1.create_window(700, 240, window=histo)
qplot=tk.Button (root, text='q plot', command=qq, bg='black',fg='white', font=('helvetica', 11, 'bold'))
canvas1.create_window(700, 280, window=qplot)
qq=tk.Button (root, text='qq plot', command=qq, bg='black',fg='white', font=('helvetica', 11, 'bold'))
canvas1.create_window(700, 320, window=qq)
scat=tk.Button (root, text='Scatter plot', command=scat, bg='black',fg='white', font=('helvetica', 11, 'bold'))
canvas1.create_window(700, 360, window=scat)
box=tk.Button (root, text='Box plot', command=box_plot, bg='black',fg='white', font=('helvetica', 11, 'bold'))
canvas1.create_window(700, 400, window=box)
graphical1=tk.Button (root, text='Correlation Analysis', bg='white',fg='black', font=('helvetica', 11, 'bold'))
canvas1.create_window(900, 190, window=graphical1)
Chi_square_test=tk.Button (root, text='Chi square test', command=chi2test_with_x_y, bg='black',fg='white', font=('helvetica', 11, 'bold'))
canvas1.create_window(900, 240, window=Chi_square_test)

Pearson_Coefficient=tk.Button (root, text='Pearson Coefficient', command=pearsoncoef_with_x_y, bg='black',fg='white', font=('helvetica', 11, 'bold'))
canvas1.create_window(900, 280, window=Pearson_Coefficient)

Covariance=tk.Button (root, text='Covariance', command=covariance_with_x_y, bg='black',fg='white', font=('helvetica', 11, 'bold'))
canvas1.create_window(900, 320, window=Covariance)


graphical2=tk.Button (root, text='Normalisation', bg='white',fg='black', font=('helvetica', 11, 'bold'))
canvas1.create_window(1100, 190, window=graphical2)
Min_Max_Normalization=tk.Button (root, text='Min-Max Normalization', command=min_max_norm_x, bg='black',fg='white', font=('helvetica', 11, 'bold'))
canvas1.create_window(1100, 240, window=Min_Max_Normalization)
Z_score_normalization=tk.Button (root, text=' Z-score normalization', command=z_score_norm_x, bg='black',fg='white', font=('helvetica', 11, 'bold'))
canvas1.create_window(1100, 280, window=Z_score_normalization)
Normalization_by_decimal_scaling=tk.Button (root, text='Normalization by decimal scaling', command=dec_scale_norm_x, bg='black',fg='white', font=('helvetica', 11, 'bold'))
canvas1.create_window(1100, 320, window=Normalization_by_decimal_scaling)

root.t1=Entry()
root.t1.place(x=200,y=220)
root.t2=Entry()
root.t2.place(x=200,y=260)
root.t3=Entry()
root.t3.place(x=200,y=300)
root.t4=Entry()
root.t4.place(x=200,y=340)
root.t5=Entry()
root.t5.place(x=200,y=380)
root.t6=Entry()
root.t6.place(x=200,y=420)
root.t7=Entry()
root.t7.place(x=500,y=220)
root.t8=Entry()
root.t8.place(x=500,y=260)
root.t9=Entry()
root.t9.place(x=500,y=300)

root.mainloop()