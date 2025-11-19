"""
https://oai.azure.com/portal/be5567c3dd4d49eb93f58914cccf3f02/deployment
clausa gpt4
"""

import time
import requests
import config
import string
import re
import sys

# returns 0 for A, 1 for B, etc, -1 for no match
def clean_output(pred):
    if not isinstance(pred, str):
        raise ValueError("Prediction must be a string.")

    pred = pred.strip()

    # match 'Answer: <LETTER>'
    match = re.search(r"Answer:\s*([A-Da-d])\s*$", pred)
    if not match:
        return -1

    letter = match.group(1).upper()
    index = ord(letter) - ord('A')
    return index

def parse_sectioned_prompt(s):

    result = {}
    current_header = None

    for line in s.split('\n'):
        if line.startswith('# '):
            # first word without punctuation
            current_header = line[2:].strip().lower().split()[0]
            current_header = current_header.translate(str.maketrans('', '', string.punctuation))
            result[current_header] = ''
        elif current_header is not None:
            result[current_header] += line + '\n'

    return result


def chatgpt(prompt, model="gpt-4o-mini", temperature=1, n=1, top_p=1, stop=None, max_tokens=10000, 
                  presence_penalty=0, frequency_penalty=0, logit_bias={}, timeout=60, log_path = None):
    messages = [{"role": "user", "content": prompt}]
    if "gpt-5" not in model:
        payload = {
            "messages": messages,
            "model": model,
            "temperature": temperature,
            "n": n,
            "top_p": top_p,
            "stop": stop,
            "max_tokens": max_tokens,
            "presence_penalty": presence_penalty,
            "frequency_penalty": frequency_penalty,
            "logit_bias": logit_bias
        }
    else:
        payload = {
            "messages": messages,
            "model": model
        }

    def log(msg: str):
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{ts}] [chatgpt] {msg}"
        if log_path is not None:
            try:
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(line + "\n")
            except Exception:
                # fall back to stderr if file logging fails
                print(line, file=sys.stderr)
        else:
            print(line, file=sys.stderr)

    retries = 0
    while True:
        try:
            r = requests.post('https://api.openai.com/v1/chat/completions',
                headers = {
                    "Authorization": f"Bearer {config.OPENAI_KEY}",
                    "Content-Type": "application/json"
                },
                json = payload,
                timeout=timeout
            )
            if r.status_code != 200:
                log(f"Non-200 status code: {r.status_code}, response: {r.text}")
                retries += 1
                time.sleep(1)
            else:
                break
        except requests.exceptions.ReadTimeout as e:
            log(f"Unexpected exception while calling OpenAI: {repr(e)}")
            time.sleep(1)
            retries += 1
        if retries > 5:
            log("Exceeded maximum retries (5). Returning [''].")
            return [""]
    try:
        r_json = r.json()
    except ValueError as e:
        log(f"Failed to parse JSON from response: {repr(e)}, raw text: {r.text}")
        return [""]

    if "choices" not in r_json:
        log(f"Response JSON missing 'choices': {r_json}")
        return [""]
    return [choice['message']['content'] for choice in r_json['choices']]


def instructGPT_logprobs(prompt, temperature=0.7):
    payload = {
        "prompt": prompt,
        "model": "text-davinci-003",
        "temperature": temperature,
        "max_tokens": 1,
        "logprobs": 1,
        "echo": True
    }
    while True:
        try:
            r = requests.post('https://api.openai.com/v1/completions',
                headers = {
                    "Authorization": f"Bearer {config.OPENAI_KEY}",
                    "Content-Type": "application/json"
                },
                json = payload,
                timeout=10
            )  
            if r.status_code != 200:
                time.sleep(2)
                retries += 1
            else:
                break
        except requests.exceptions.ReadTimeout:
            time.sleep(5)
    r = r.json()
    return r['choices']


