import requests
import json
import concurrent.futures
from abc import ABC, abstractmethod
from typing import List, Dict, Callable
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report
from utils import clean_output

class DataProcessor(ABC):
    def __init__(self, data_dir, max_threads=1):
        self.data_dir = data_dir
        self.max_threads = max_threads

    @abstractmethod
    def get_train_examples(self):
        pass

    @abstractmethod
    def get_test_examples(self):
        pass

    @abstractmethod
    def evaluate(self, predictor, test_exs):
        pass

    @abstractmethod
    def stringify_prediction(self, pred):
        pass




def process_example(ex, predictor, prompt):
    try:
        pred = predictor.inference(ex, prompt)
        return ex, pred
    except Exception as e:
        # return a sentinel so the caller can skip it
        return ex, None


class ClassificationTask(DataProcessor):

    def run_evaluate(self, predictor, prompt, test_exs, n=100):
        labels = []
        preds = []
        texts = []
        responses = []
        with concurrent.futures.ProcessPoolExecutor(max_workers=self.max_threads) as executor:
            futures = [executor.submit(process_example, ex, predictor, prompt) for ex in test_exs[:n]]
            for i, future in tqdm(enumerate(concurrent.futures.as_completed(futures)), total=len(futures), desc='running evaluate'):
                ex, pred = future.result()
                texts.append(ex['text'])
                labels.append(ex['label'])
                responses.append(pred)
                preds.append(1 if pred.strip().upper().endswith("{LABEL : YES}") else 0)

        accuracy = accuracy_score(labels, preds)
        print("accuracy:", accuracy)
        f1 = f1_score(labels, preds, average='micro')
        return f1, texts, labels, preds, responses

    def evaluate(self, predictor, prompt, test_exs, n=100):
        while True:
            try:
                f1, texts, labels, preds, responses = self.run_evaluate(predictor, prompt, test_exs, n=n)
                break
            except (concurrent.futures.process.BrokenProcessPool, requests.exceptions.SSLError):
                pass
        return f1, texts, labels, preds, responses


class BinaryClassificationTask(ClassificationTask):
    categories = ['No', 'Yes']

    def stringify_prediction(self, pred):
        return BinaryClassificationTask.categories[pred]


class EthosBinaryTask(BinaryClassificationTask):
    categories = ['No', 'Yes']

    def get_train_examples(self):
        df = pd.read_csv(self.data_dir + '/ethos_ishate_binary_shuf.csv', sep=';', header=None)
        df = df[(df[1] <= 0) | (df[1] >= 0.7)]
        exs = df.reset_index().to_dict('records')
        exs = [{'id': x['index'], 'text': x[0], 'label': 1 if x[1] > 0.4 else 0} for x in exs[200:]]
        return exs
    
    def get_test_examples(self):
        df = pd.read_csv(self.data_dir + '/ethos_ishate_binary_shuf.csv', sep=';', header=None)
        df = df[(df[1] <= 0) | (df[1] >= 0.7)]
        exs = df.reset_index().to_dict('records')
        exs = [{'id': x['index'], 'text': x[0], 'label': 1 if x[1] > 0.4 else 0} for x in exs[:200]]
        return exs


class JailbreakBinaryTask(BinaryClassificationTask):
    categories = ['No', 'Yes']

    def get_train_examples(self):
        exs = []
        for i, l in enumerate(open(self.data_dir + '/train.tsv')):
            convo, label = l.strip().split('\t')
            label = int(label)
            text = ' '.join([x['text'].strip() for x in json.loads(convo) if x['role'] == 'user'])
            exs.append({'id': i, 'text': text, 'label': label})
        return exs
    
    def get_test_examples(self):
        exs = []
        for i, l in enumerate(open(self.data_dir + '/test.tsv')):
            convo, label = l.strip().split('\t')
            label = int(label)
            text = ' '.join([x['text'].strip() for x in json.loads(convo) if x['role'] == 'user'])
            exs.append({'id': i, 'text': text, 'label': label})
        return exs


class DefaultHFBinaryTask(BinaryClassificationTask):
    categories = ['No', 'Yes']

    def get_train_examples(self):
        exs = []
        for i, row in enumerate(open(self.data_dir + '/train.jsonl')):
            row = json.loads(row.strip())
            exs.append({'id': f'train-{i}', 'label': row['label'], 'text': row['text']})
        return exs
    
    def get_test_examples(self):
        exs = []
        for i, row in enumerate(open(self.data_dir + '/test.jsonl')):
            row = json.loads(row.strip())
            exs.append({'id': f'test-{i}', 'label': row['label'], 'text': row['text']})
        return exs

class MMLUTask(ClassificationTask):
    categories = ['A', 'B', 'C', 'D']

    def stringify_prediction(self, pred):
        return MMLUTask.categories[pred]

    def get_train_examples(self):
        exs = []
        for i, row in enumerate(open(self.subject_dir + '/train.jsonl')):
            row = json.loads(row.strip())
            exs.append({'id': f'train-{i}', 'label': row['label'], 'text': row['text'], 'choices': row['choices']})
        return exs
    
    def get_test_examples(self):
        exs = []
        for i, row in enumerate(open(self.subject_dir + '/test.jsonl')):
            row = json.loads(row.strip())
            exs.append({'id': f'test-{i}', 'label': row['label'], 'text': row['text'], 'choices': row['choices']})
        return exs

    def run_evaluate(self, predictor, prompt, test_exs):
        labels = []
        preds = []
        texts = []
        choices = []
        responses = []
        with concurrent.futures.ProcessPoolExecutor(max_workers=self.max_threads) as executor:
            futures = [executor.submit(process_example, ex, predictor, prompt) for ex in test_exs]  # processes all examples now
            for i, future in tqdm(enumerate(concurrent.futures.as_completed(futures)), total=len(futures), desc='running evaluate'):
                ex, pred = future.result()
                if pred is None:
                    continue  # skip this example due to inference error
                texts.append(ex['text'])
                choices.append(ex['choices'])
                labels.append(ex['label'])
                responses.append(pred)
                preds.append(clean_output(pred))

        accuracy = accuracy_score(labels, preds)
        f1 = f1_score(labels, preds, average='micro')
        return f1, accuracy, texts, labels, choices, preds, responses

    def evaluate(self, predictor, prompt, test_exs, n):
        while True:
            try:
                f1, accuracy, texts, labels, choices, preds, responses = self.run_evaluate(predictor, prompt, test_exs)
                break
            except (concurrent.futures.process.BrokenProcessPool, requests.exceptions.SSLError):
                pass
        return f1, accuracy, texts, labels, choices, preds, responses

    def evaluate_on_all_subjects(self, prompt, predictor, subjects):
        """
        Evaluate a single prompt on ALL MMLU subjects.
        Returns: dict {subject_name: accuracy}
        """
        results = {}
        original_dir = self.subject_dir  # to be restored

        for subj in subjects:
            self.subject_dir = f"{self.data_dir}/{subj}"
            test_exs = self.get_test_examples()

            f1, accuracy, texts, labels, choices, preds, responses = self.evaluate(
                predictor, prompt, test_exs, n=len(test_exs)
            )
            results[subj] = accuracy

        # restore original subject dir
        self.subject_dir = original_dir
        return results