from flask import Flask, render_template, request, jsonify
import requests
import io

app = Flask(__name__)

AZURE_ENDPOINT = "https://formtestlsw.cognitiveservices.azure.com/formrecognizer/v2.1/prebuilt/receipt/analyze"
AZURE_HEADERS = {
    "Ocp-Apim-Subscription-Key": "2fe1b91a80f94bb2a751f7880f00adf6",
    "Content-Type": "application/pdf"
}

@app.route("/", methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            img_bytes = file.read()

            response = requests.post(AZURE_ENDPOINT, headers=AZURE_HEADERS, data=img_bytes)
            if response.status_code == 202:
                operation_location = response.headers.get("Operation-Location")
                return jsonify({"status": "processing", "operation_location": operation_location})
            elif response.status_code == 200:
                return jsonify(response.json())
            else:
                return jsonify({"error": f"Error from Azure Form Recognizer API: {response.status_code} - {response.text}"})

    return '''
    <!doctype html>
    <title>Upload PDF to Azure Form Recognizer</title>
    <h1>Upload PDF</h1>
    <form action="" method=post enctype=multipart/form-data>
      <input type=file name=file>
      <input type=submit value=Upload>
    </form>
    '''

if __name__ == "__main__":
    app.run(debug=True)
