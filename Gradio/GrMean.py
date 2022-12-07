import gradio as gr
import pandas as pd
import numpy as np

df = []
def update(file):
    global df
    df = pd.read_csv(file.name)
    df.fillna(0, inplace=True)
    if len(df)>2:
          data = df
          cols=[]
          for i in data.columns[:-1]:
              cols.append(i)
          sum = 0
          arr=[]
          colnm = []
          res=[]
          dmean = dict()
          dmedian = dict()
          dstd = dict()
          dvar = dict()
          dmode = dict()
          for i in cols:
              colnm.append(str(i))


          for attribute1 in cols:
            freq = {}
            for i in range(len(data)):
                freq[data.loc[i, attribute1]] = 0
            maxFreq = 0
            maxFreqElem = 0              
          
            # attribute1=cols[0]
            for i in range(len(data)):
                sum += data.loc[i, attribute1]
                arr.append(data.loc[i, attribute1])
                freq[data.loc[i, attribute1]] = freq[data.loc[i, attribute1]]+1
                if freq[data.loc[i, attribute1]] > maxFreq:
                    maxFreq = freq[data.loc[i, attribute1]]
                    maxFreqElem = data.loc[i, attribute1]
            avg=np.mean(arr)
            # res.append("Mean of attribute (" + attribute1 + ") is " + str(avg))
            dmedian[attribute1]=[np.median(arr)]
            dmean[attribute1]=[avg]
            dvar[attribute1] = [np.var(arr)]
            dstd[attribute1]=[np.std(arr)]
            dmode[attribute1]=[maxFreqElem]
          # printf(res)
          mean = pd.DataFrame(data=dmean)
          median = pd.DataFrame(data=dmedian)
          variance = pd.DataFrame(data=dvar)
          std = pd.DataFrame(data=dstd)
          mode = pd.DataFrame(data=dmode)
          # res = pd.DataFrame(columns=res)
    return mean, mode, median, variance, std,df

with gr.Blocks() as demo:
    gr.Markdown(" <center><h1>Data Analysis Using Gradio</h1> </center>")
    
    with gr.Row():
        inp = gr.File(label="Upload file")
    btn = gr.Button("Run")
    
    with gr.Column():
        out1 = gr.DataFrame(label="Mean", interactive=1)
        out2 = gr.DataFrame(label="Mode", interactive=1)
        out3 = gr.DataFrame(label="Median", interactive=1)
        out4 = gr.DataFrame(label="Variance", interactive=1)
        out5 = gr.DataFrame(label="Standard Deviation", interactive=1)
    with gr.Column():
        out = gr.DataFrame(row_count = (2, "dynamic"), col_count=(4,"dynamic"), label="Input Data", interactive=1)
    btn.click(fn=update, inputs=inp, outputs=[out1,out2, out3, out4, out5,out])
    

demo.launch(debug=True)