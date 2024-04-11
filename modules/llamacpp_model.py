import re
from functools import partial

import numpy as np
import torch

from modules import RoPE, llama_cpp_python_hijack, shared
from modules.callbacks import Iteratorize
from modules.logging_colors import logger
from modules.text_generation import get_max_prompt_length


import llama_cpp_cuda

class LlamaCppModel:
    def __init__(self):
        self.initialized = False

    def __del__(self):
        del self.model

    @classmethod
    def from_pretrained(cls, model_name):

        result = cls()

        path = Path(f'app/models/{model_name}')

        params = {
            'model_path': str(path),
            'n_ctx': 1024,
            'n_gpu_layers': -1,
        }

        result.model = llama_cpp_cuda.Llama(**params)

        return result


    def generate(self, prompt, state, callback=None):
        LogitsProcessorList = llama_cpp_lib().LogitsProcessorList
        prompt = prompt if type(prompt) is str else prompt.decode()

        self.load_grammar(state['grammar_string'])
        logit_processors = LogitsProcessorList()
        if state['ban_eos_token']:
            logit_processors.append(partial(ban_eos_logits_processor, self.model.token_eos()))

        if state['custom_token_bans']:
            to_ban = [int(x) for x in state['custom_token_bans'].split(',')]
            if len(to_ban) > 0:
                logit_processors.append(partial(custom_token_ban_logits_processor, to_ban))

        completion_chunks = self.model.create_completion(
            prompt=prompt,
            max_tokens=state['max_new_tokens'],
            temperature=state['temperature'],
            top_p=state['top_p'],
            min_p=state['min_p'],
            typical_p=state['typical_p'],
            frequency_penalty=state['frequency_penalty'],
            presence_penalty=state['presence_penalty'],
            repeat_penalty=state['repetition_penalty'],
            top_k=state['top_k'],
            stream=True,
            seed=int(state['seed']) if state['seed'] != -1 else None,
            tfs_z=state['tfs'],
            mirostat_mode=int(state['mirostat_mode']),
            mirostat_tau=state['mirostat_tau'],
            mirostat_eta=state['mirostat_eta'],
            logits_processor=logit_processors,
            grammar=self.grammar
        )

        output = ""
        for completion_chunk in completion_chunks:
            if shared.stop_everything:
                break

            text = completion_chunk['choices'][0]['text']
            output += text
            if callback:
                callback(text)

        return output

    def generate_with_streaming(self, *args, **kwargs):
        with Iteratorize(self.generate, args, kwargs, callback=None) as generator:
            reply = ''
            for token in generator:
                reply += token
                yield reply
