
import tkinter as tk
import csv
from collections import Counter
from tkinter import filedialog, messagebox, ttk, StringVar, OptionMenu, Button, Label

import pandas as pd
import matplotlib.pyplot as plt

# initalise the tkinter GUI
root = tk.Tk()
root.title("Data Mining : Assignment 1")

root.geometry("1600x1000") # set the root dimensions
root.configure(bg='palegreen')
root.pack_propagate(False) # tells the root to not let the widgets inside it determine its size.
root.resizable(0, 0) # makes the root window fixed in size.

# Frame for TreeView
frame1 = tk.LabelFrame(root, text="Excel Data")
frame1.place(height=550, width=1600)

# Frame for open file dialog
file_frame = tk.LabelFrame(root, text="Open File")
file_frame.place(height=100, width=400, rely=0.65, relx=0)

# Buttons
button1 = tk.Button(file_frame, text="Browse A File", command=lambda: File_dialog())
button1.place(rely=0.65, relx=0.50)

button2 = tk.Button(file_frame, text="Load File", command=lambda: Load_excel_data())
button2.place(rely=0.65, relx=0.30)

# The file/file path text
label_file = ttk.Label(file_frame, text="No File Selected")
label_file.place(rely=0, relx=0)


## Treeview Widget
tv1 = ttk.Treeview(frame1)
tv1.place(relheight=1, relwidth=1) # set the height and width of the widget to 100% of its container (frame1).

treescrolly = tk.Scrollbar(frame1, orient="vertical", command=tv1.yview) # command means update the yaxis view of the widget
treescrollx = tk.Scrollbar(frame1, orient="horizontal", command=tv1.xview) # command means update the xaxis view of the widget
tv1.configure(xscrollcommand=treescrollx.set, yscrollcommand=treescrolly.set) # assign the scrollbars to the Treeview Widget
treescrollx.pack(side="bottom", fill="x") # make the scrollbar fill the x axis of the Treeview widget
treescrolly.pack(side="right", fill="y") # make the scrollbar fill the y axis of the Treeview widget

clicked = StringVar()
clicked.set( "Select Column" )
label = Label( root , text = "" )
label.place(rely=0.72, relx=0.40)
medianlib = Label(root, text="")
medianlib.place(rely=0.75,relx=0.40)
mean = Label( root , text = "Mean:" )
mean.place(rely=0.72, relx=0.35)
median = Label( root, text = "Median:")
median.place(rely=0.75, relx=0.35)
mode = Label(root, text="Mode:")
mode.place(rely=0.78, relx=0.35)
modelib = Label(root, text="")
modelib.place(rely=0.78, relx=0.40)
midrange = Label(root, text="Mid-Range:")
midrange.place(rely=0.81, relx=0.35)
variance= Label(root, text="Variance:")
variance.place(rely=0.84, relx=0.35)
stddev = Label(root, text="Std Deviation:")
stddev.place(rely=0.87, relx=0.35)
range = Label( root , text = "Range:" )
range.place(rely=0.72, relx=0.55)
quartile = Label( root, text = "quartile:")
quartile.place(rely=0.75, relx=0.55)
interquartile = Label(root, text="Inter quartile range:")
interquartile.place(rely=0.78, relx=0.55)
fns = Label(root, text="Five number summary:")
fns.place(rely=0.81, relx=0.55)

df=[]

def File_dialog():
    """This Function will open the file explorer and assign the chosen file path to label_file"""
    filename = filedialog.askopenfilename(initialdir="/home/DM/Assigment2",
                                          title="Select A File")
    label_file["text"] = filename
    return None


def Load_excel_data():
    """If the file selected is valid this will load the file into the Treeview"""
    file_path = label_file["text"]
    
    try:
        excel_filename = r"{}".format(file_path)
        if excel_filename[-4:] == ".csv":
            df = pd.read_csv(excel_filename)
        else:
            df = pd.read_excel(excel_filename)

    except ValueError:
        tk.messagebox.showerror("Information", "The file you have chosen is invalid")
        return None
    except FileNotFoundError:
        tk.messagebox.showerror("Information", f"No such file as {file_path}")
        return None

    clear_data()
    tv1["column"] = list(df.columns)
    tv1["show"] = "headings"
    for column in tv1["columns"]:
        tv1.heading(column, text=column) # let the column heading = column name

    df_rows = df.to_numpy().tolist() # turns the dataframe into a list of lists
    for row in df_rows:
        tv1.insert("", "end", values=row) # inserts each list into the treeview. For parameters see https://docs.python.org/3/library/tkinter.ttk.html#tkinter.ttk.Treeview.insert
    
    # Create Dropdown menu
    drop = OptionMenu( root , clicked , *tv1["column"] )
    drop.place(rely=0.65, relx=0.50)
    
    # Create button, it will change label text
    button = Button( root , text = "click Me" , command = lambda: show(df)).place(rely=0.68, relx=0.50)
    #print(df['UDI'].mean())
    return None


def clear_data():
    tv1.delete(*tv1.get_children())
    return None



def show(df):
    selectedRow=clicked.get()
    sum=0
    for x in df[selectedRow]:
        sum+=x
    label1 = Label( root , text = sum/len(df[selectedRow]) )
    label1.place(rely=0.72, relx=0.45)
    label.config( text = df[selectedRow].mean() )
    sortedLst = sorted(df[selectedRow])
    lstLen = len(df[selectedRow])
    index = (lstLen - 1) // 2
    medianlib.config(text=df[selectedRow].median())
    if (lstLen % 2):
        Medianval = Label( root , text = sortedLst[index] )
        Medianval.place(rely=0.75, relx=0.45)
    else:
        Medianval = Label( root , text = (sortedLst[index] + sortedLst[index + 1])/2.0 )
        Medianval.place(rely=0.75, relx=0.45)
    frequency = {}

    for value in df[selectedRow]:
        # print(value)
        frequency[value] = frequency.get(value, 0) + 1

    most_frequent = max(frequency.values())

    modes = [key for key, value in frequency.items()
                      if value == most_frequent]
    modeval= Label(root, text=modes )
    modeval.place(rely=0.78 , relx=0.45)
    modelib.config(text=df[selectedRow].mode()[0])
    midrangeval=Label(root, text=(max(df[selectedRow]-min(df[selectedRow]))/2))
    midrangeval.place(rely=0.81, relx=0.40)
    variancelib=Label(root, text=df[selectedRow].var())
    variancelib.place(rely=0.84,relx=0.40)
    stddevlib=Label(root, text=df[selectedRow].std())
    stddevlib.place(rely=0.87, relx=0.40)
    # print(sum/len(df[selectedRow]))
    # print(df[selectedRow][0])
    df[selectedRow].hist(bins=8)
    # df[selectedRow].plot.scatter(x="side1", y="side2")
    df.boxplot(by=selectedRow)
    plt.savefig("hist_01.png", bbox_inches='tight', dpi=100)
root.mainloop()