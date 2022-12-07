# Import module
from distutils import ccompiler
from tkinter import *

def findMean():
    print("mean")


# Create object
root = Tk()

# Adjust size
root.geometry( "200x200" )

# Change the label text
def show():
	text = clicked.get()
    if text == "Mean":
        

# Dropdown menu options
options = [
	"Mean",
	"Median",
	"Mode",
	"Thursday",
	"Friday",
	"Saturday",
	"Sunday"
]

# datatype of menu text
clicked = StringVar()

# initial menu text
clicked.set( "Measure Of Central Tendancy" )

# Create Dropdown menu
drop = OptionMenu( root , clicked , *options )
drop.place(x=10,y=10)
#drop.pack()

# Create button, it will change label text
button = Button( root , text = "click Me" , command = show ).pack()

# Create Label
label = Label( root , text = " " )
label.pack()

# Execute tkinter
root.mainloop()
