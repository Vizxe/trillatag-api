import os
import accelerate

from flask import Flask, request
from modules.download_model import download
from modules.llamacpp_model import LlamaCppModel

app = Flask(__name__)
model = None


@app.route('/')
def index():
    global model
    if model is None:
        model_path = download(repo_id="TrillaBit/TrillaTag-0.0.7", filename="trillatag-0.0.7.Q5_K_M.gguf")
        model = LlamaCppModel.load(model_path)

    prompt = request.args.get('prompt', default='No prompt')

    output = ""
    for c in model.generate(prompt):
        output += c

    return f"Prompt: {prompt}\nOutput: {output}"


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7860)
