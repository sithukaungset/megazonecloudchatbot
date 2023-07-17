# this is for hugging face
import json
import requests
API_URL = "https://api-inference.huggingface.co/models/gpt3"
headers = {"Authorization": f"Bearer {API_TOKEN}"}
def query(payload):
	data = json.dumps(payload)
	response = requests.request("POST", API_URL, headers= headers, data=data)
	
	return json.loads(response.content.decode("utf-8"))
data = query("Can you please let us know more details about your ")

import gradio as gr

def greet(name):
	return "Hello" + name + "!"

demo = gr.Interface(fn= greet, inputs="text", outputs="text")
# to launch the demo 
demo.launch()


 
