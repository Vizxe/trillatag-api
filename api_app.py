from modules.text_generation import generate_reply_wrapper
from modules.utils import gradio

from flask import Flask, request

app = Flask(__name__)

@app.route('/')
def index():
    state = gradio('interface_state')
    prompt = request.args.get('prompt')
    ans = ""
    for token in generate_reply_wrapper(prompt, state):
        ans += token
    return ans

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)