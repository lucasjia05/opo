from abc import ABC, abstractmethod
from typing import List, Dict, Callable
from liquid import Template

import utils
import tasks

class GPT4Predictor(ABC):
    def __init__(self, opt):
        self.opt = opt

    @abstractmethod
    def inference(self, ex, prompt):
        pass

class BinaryPredictor(GPT4Predictor):
    categories = ['No', 'Yes']

    # def inference(self, ex, prompt):
    #     prompt = Template(prompt).render(text=ex['text'])
    #     response = utils.chatgpt(
    #         prompt, max_tokens=4, n=1, timeout=30, 
    #         temperature=self.opt['temperature'], model=self.opt['task_model'])[0]
    #     pred = 1 if response.strip().upper().startswith('YES') else 0
    #     return pred

    def inference(self, ex, prompt):
        prompt = Template(prompt).render(text=ex['text'])
        response = utils.chatgpt(
            prompt, max_tokens=4096, n=1, timeout=180, 
            temperature=self.opt['temperature'], model=self.opt['task_model'])[0]
        # pred = 1 if "{LABEL : YES}" in response.strip().upper() else 0
        # pred = 1 if response.strip().upper().endswith("{LABEL : YES}") else 0
        return response

class MMLUPredictor(GPT4Predictor):
    def inference(self, ex, prompt):
        formatted_choices = "\n".join(
            [f"({chr(65+i)}) {choice}" for i, choice in enumerate(ex['choices'])]
        )
        prompt = Template(prompt).render(
            text=ex['text'], 
            choices=formatted_choices
        )
        #print(prompt)
        response = utils.chatgpt(
            prompt, max_tokens=4096, n=1, timeout=180, 
            temperature=self.opt['temperature'], model=self.opt['task_model'])[0]
        #print("response: ", response)
        return response