# this is for hugging face
import json
import requests
API_URL = "https://api-inference.huggingface.co/models/gpt3"
headers = {"Authorization": f"Bearer {API_TOKEN}"}
def query(payload):
	data = json.dumps(payload)
	response = requests.request("POST", API_URL, header)

