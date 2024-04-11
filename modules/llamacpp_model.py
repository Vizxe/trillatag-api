import numpy as np
import torch
import llama_cpp_cuda


class LlamaCppModel:
    def __init__(self):
        ...

    def __del__(self):
        del self.model

    @classmethod
    def load(cls, path):
        result = cls()

        params = {
            'model_path': path,
            'n_ctx': 1024,
            'n_gpu_layers': -1,
        }

        result.model = llama_cpp_cuda.Llama(**params)

        return result

    def generate(self, prompt):
        prompt = prompt if type(prompt) is str else prompt.decode()

        completion_chunks = self.model.create_completion(
            prompt=prompt,
            max_tokens=1024,
            n_gpu_layers=-1,
            stream=True,
            stop=["[END]"]
        )

        output = ""
        for completion_chunk in completion_chunks:
            text = completion_chunk['choices'][0]['text']
            output += text
            yield output

        return output
