import gradio as gr
import pandas as pd

dataset = []
def load_file(filename):
    data = pd.read_csv(filename.name)
    global dataset
    dataset = pd.DataFrame(data)
    return data

with gr.Blocks() as demo:
    with gr.Tab("Display Dataset"):
        df = gr.File(label="Upload Dataset")
        output = gr.DataFrame(label="Dataset")
        btn = gr.Button("Submit")
        # dataset = load_file(df)
        btn.click(fn=load_file, inputs=df, outputs=output)

    # with gr.Tab("Exploratory Analysis"):
    #     if gr.Button().click():
    #         print(dataset)
        
demo.launch(debug=True)