import numpy as np
from tqdm import tqdm
import random
from abc import ABC, abstractmethod
import utils

class PromptOptimizer(ABC):
    def __init__(self, args, evaluator_fn, scorer, max_threads=1, bf_eval=None):
        self.opt = args
        self.evaluator_fn = evaluator_fn
        self.scorer = scorer
        self.max_threads = max_threads
        self.bf_eval = bf_eval
        self.metrics = {"acc" : [], "f1": []}

    @abstractmethod
    def expand_candidates(self, prompts, task, gpt4, train_exs):
        pass

class ProTeGi(PromptOptimizer):
    """ ProTeGi: Prompt Optimization with Textual Gradients
    """
    def _sample_error_str(self, texts, labels, preds, task, n=4):
        """ Sample n error strings from the given texts, labels, and preds"""
        error_idxs = []
        for i, (l, p) in enumerate(zip(labels, preds)):
            if l != p:
                error_idxs.append(i)

        sample_idxs = random.sample(error_idxs, min(len(error_idxs), n))

        sample_texts = [texts[i] for i in sample_idxs]
        sample_labels = [labels[i] for i in sample_idxs]
        sample_preds = [preds[i] for i in sample_idxs]
        error_string = ''
        num_errors = 0
        error_idx = 0
        for i, (t, l, p) in enumerate(zip(sample_texts, sample_labels, sample_preds)):
            error_string += f'## Example {error_idx+1}\n'
            error_string += f'Text: \"{t.strip()}\"\nLabel: {task.stringify_prediction(l)}\nPrediction: {task.stringify_prediction(p)}\n\n'
            error_idx += 1
        return error_string.strip()

    def parse_tagged_text(self, text, start_tag, end_tag):
        """ Parse text that is tagged with start and end tags."""
        texts = []
        while True:
            start_index = text.find(start_tag)
            if start_index == -1:
                break
            end_index = text.find(end_tag, start_index)
            if end_index == -1:
                break
            start_index += len(start_tag)
            texts.append(text[start_index:end_index].strip())
            text = text[end_index+len(end_tag):]
        return texts

    def _get_gradients(self, prompt, error_string, num_feedbacks=5, n=1):
        """ Get "gradients" for a prompt based on the error string."""
        gradient_prompt = f"""
        I'm trying to write a zero-shot classifier prompt.
    
        My current prompt is:
        "{prompt}"

        But this prompt gets the following examples wrong:
        {error_string}

        give {num_feedbacks} reasons why the prompt could have gotten these examples wrong.
        Wrap each reason with <START> and <END>
        """
        gradient_prompt = '\n'.join([line.lstrip() for line in gradient_prompt.split('\n')])
        res = utils.chatgpt(gradient_prompt, n=n)
        feedbacks = []
        new_prompts = []
        for r in res:    
            feedbacks += self.parse_tagged_text(r, "<START>", "<END>")
        return feedbacks

    def apply_gradient(self, prompt, error_str, feedback_str, steps_per_gradient, n=1):
        """ Incorporate feedback gradient into a prompt."""
        transformation_prompt = f"""
        I'm trying to write a zero-shot classifier.
        
        My current prompt is:
        "{prompt}"

        But it gets the following examples wrong:
        {error_str}

        Based on these examples the problem with this prompt is that {feedback_str}

        Based on the above information, I wrote {steps_per_gradient} different improved prompts.
        Each prompt is wrapped with <START> and <END>.

        The {steps_per_gradient} new prompts are:
        """
        transformation_prompt = '\n'.join([line.lstrip() for line in transformation_prompt.split('\n')])
        res = utils.chatgpt(transformation_prompt, n=n)
        new_prompts = []
        for r in res:   
            new_prompts += self.parse_tagged_text(r, "<START>", "<END>")
        return new_prompts

    def generate_synonyms(self, prompt_section, n=3):
        """ Generate synonyms for a prompt section."""
        rewriter_prompt = f"Generate a variation of the following instruction while keeping the semantic meaning.\n\nInput: {prompt_section}\n\nOutput:"
        new_instructions = utils.chatgpt(rewriter_prompt, n=n)
        new_instructions = [x for x in new_instructions if x]
        return new_instructions

    def get_gradients(self, prompt, task_section, task, gpt4, texts, labels, preds):
        """ Get "gradients" for a prompt based on sampled error strings."""
        prompt_feedbacks = []
        for _ in tqdm(range(self.opt['n_gradients']), total=self.opt['n_gradients'], desc='gradients..'):
            error_string = self._sample_error_str(
                texts, labels, preds, task, n=self.opt['errors_per_gradient'])
            gradients = self._get_gradients(
                task_section, error_string, self.opt['gradients_per_error'], n=1)
            prompt_feedbacks += [(t, error_string) for t in gradients]
        return prompt_feedbacks

    def expand_candidates(self, prompts, task, gpt4, train_exs):
        """ Expand a list of prompts by generating gradient-based successors and 
            synonyms for each section.
        """
        minibatch = random.sample(train_exs, k=self.opt['minibatch_size'])

        new_prompts = []
        for prompt in tqdm(prompts, desc=f'expanding {len(prompts)} prompts'):
            sections = utils.parse_sectioned_prompt(prompt)
            task_section = sections['task'].strip()

            # evaluate prompt on minibatch
            _, texts, labels, preds = task.evaluate(gpt4, prompt, minibatch)

            # get gradients
            new_task_sections = []
            if self.opt['n_gradients'] > 0:
                gradients = self.get_gradients(prompt, task_section, task, gpt4, texts, labels, preds)
                new_task_sections = []
                for feedback, error_string in tqdm(gradients, desc='applying gradients'):
                    tmp = self.apply_gradient(
                        task_section, error_string, feedback, self.opt['steps_per_gradient'])
                    new_task_sections += tmp

            # generate synonyms
            mc_sampled_task_sections = []
            if self.opt['mc_samples_per_step'] > 0:
                for sect in tqdm(new_task_sections + [task_section], desc='mc samples'):
                    mc_sects = self.generate_synonyms(
                        sect, n=self.opt['mc_samples_per_step'])
                    mc_sampled_task_sections += mc_sects

            # combine
            new_sections = new_task_sections + mc_sampled_task_sections
            new_sections = list(set(new_sections)) # dedup
            tmp_new_prompts = [
                prompt.replace(task_section, tmp) 
                for tmp in new_sections
            ]
            
            # filter a little
            if len(new_sections) > self.opt['max_expansion_factor']:
                if self.opt['reject_on_errors']:
                    error_exs = []
                    for i, (t, l, p) in enumerate(zip(texts, labels, preds)):
                        if l != p:
                            error_exs.append({'text': t, 'label': l})
                    error_exs = random.sample(error_exs, min(len(error_exs), 16))

                    # speed up a little
                    tmp_new_prompts = random.sample(tmp_new_prompts, min(len(tmp_new_prompts), self.opt['max_expansion_factor'] * 2))

                    error_scores = self.bf_eval(tmp_new_prompts, error_exs, task, gpt4, self.scorer, max_threads=self.max_threads)
                    tmp_new_prompts = [tmp_new_prompts[i] for i in np.argsort(error_scores)[-self.opt['max_expansion_factor']:]]
                else:
                    tmp_new_prompts = random.sample(tmp_new_prompts, 
                        k=self.opt['max_expansion_factor'])

            new_prompts += tmp_new_prompts

        new_prompts += prompts # add originals
        new_prompts = list(set(new_prompts)) # dedup

        return new_prompts

    def score_candidates(self, prompts, task, gpt4, train_exs):
        """ Score a list of prompts."""
        if len(prompts) == 1:
            return [1.0]

        evals = self.evaluator_fn(
            prompts, train_exs, task, gpt4,
            scorer=self.scorer,
            rounds=self.opt['eval_rounds'],
            num_prompts_per_round=self.opt['eval_prompts_per_round'],
            samples_per_eval=self.opt['samples_per_eval'],
            max_threads=self.max_threads
        )
        return evals


class OnlineProTeGi(PromptOptimizer):
    """ ProTeGi: Prompt Optimization with Textual Gradients
    """
    def __init__(self, args, evaluator_fn, scorer, max_threads=1, bf_eval=None):
        self.opt = args
        self.evaluator_fn = evaluator_fn
        self.scorer = scorer
        self.max_threads = max_threads
        self.bf_eval = bf_eval
        self.metrics = {"acc" : [], "f1": []}
        self.best_prompt = "N/A"
        self.best_score = -1

    def _sample_error_str(self, texts, choices, labels, preds, responses, task, n=4):
        """ Sample n error strings from the given texts, labels, and preds"""
        error_idxs = []
        for i, (l, p) in enumerate(zip(labels, preds)):
            if l != p:
                error_idxs.append(i)

        acc =  1 - len(error_idxs) / len(labels)
        self.metrics["acc"].append(acc)
        self.metrics['avg_acc'] = sum(self.metrics["acc"]) / len(self.metrics["acc"])
        
        sample_idxs = random.sample(error_idxs, min(len(error_idxs), n))

        sample_texts = [texts[i] for i in sample_idxs]
        sample_choices = [choices[i] for i in sample_idxs]
        sample_labels = [labels[i] for i in sample_idxs]
        sample_preds = [preds[i] for i in sample_idxs]
        sample_resps = [responses[i] for i in sample_idxs]
        error_string = ''
        num_errors = 0
        error_idx = 0
        for i, (t, c, l, p, r) in enumerate(zip(sample_texts, sample_choices, sample_labels, sample_preds, sample_resps)):
            error_string += f'## Example {error_idx+1}\n'
            error_string += f'Text: \"{t.strip()}\"\n'
            error_string += "Choices:\n" + "\n".join(
                f"({chr(65 + i)}) {opt}" for i, opt in enumerate(c)
            ) + "\n"
            error_string += f'Correct Answer: {task.stringify_prediction(l)}\n'
            error_string += f'Response: {r.strip()}\n\n'
            error_idx += 1
        return error_string.strip()

    def parse_tagged_text(self, text, start_tag, end_tag):
        """ Parse text that is tagged with start and end tags."""
        texts = []
        text = text.replace("</END>", "<END>").replace("</end>", "<END>").replace("</End>", "<END>")    # added to fix </END> issues
        while True:
            start_index = text.find(start_tag)
            if start_index == -1:
                break
            end_index = text.find(end_tag, start_index)
            if end_index == -1:
                break
            start_index += len(start_tag)
            texts.append(text[start_index:end_index].strip())
            text = text[end_index+len(end_tag):]
        return texts

    def _get_gradients(self, prompt, error_string, num_feedbacks=1, n=1, model="gpt-4o"):
        """ Get "gradients" for a prompt based on the error string."""
        acc = self.metrics["acc"][-1]
        overall_acc = sum(self.metrics["acc"]) / len(self.metrics["acc"])
        gradient_prompt = f"""
        I'm trying to write a multiple-choice question answering prompt across different subject areas.

        Here is the best performing prompt so far:
        "{self.best_prompt}"

        Here is the best prompt's accuracy:
        {self.best_score}
    
        My current prompt is:
        "{prompt}"

        Here is the current prompt's accuracy:
        {acc}

        But this prompt gets the following examples wrong:
        {error_string}

        Carefully analyze these mistakes step by step.

        For each example:
        1. Restate in your own words what the question is asking.
        2. Identify which option is correct and briefly explain why.
        3. Describe why the model’s chosen answer is wrong or incomplete.
        4. Explain how the wording or structure of the CURRENT PROMPT could have encouraged this mistake

        After you have thought through all of the examples in detail, synthesize your analysis into a single, concise description of the main weakness or set of weaknesses in the current prompt.
        Focus specifically on how the prompt is guiding the model's reasoning and behavior, rather than generic statements.

        Wrap ONLY this final synthesized description with <START> and <END>.
        Do not include <START> or <END> anywhere else in your response.
        Do NOT propose a new prompt here; just diagnose the issues with the existing one.
        """
        gradient_prompt = '\n'.join([line.lstrip() for line in gradient_prompt.split('\n')])
        res = utils.chatgpt(gradient_prompt, n=n, model=model)
        feedbacks = []
        new_prompts = []
        # with open(self.opt['out'], 'a') as outf:
        #     outf.write(f"errors: {error_string}\n")
        with open(self.opt['logs'], 'a') as outf:
            #outf.write(f"error string: {error_string}\n")
            outf.write("---------------------get gradients---------------\n")
            outf.write(f"prompt: {gradient_prompt}\n")
            outf.write(f"gradients: {res}\n")
        for r in res:    
            feedbacks += self.parse_tagged_text(r, "<START>", "<END>")
        return feedbacks

    def apply_gradient(self, prompt, error_str, feedback_str, steps_per_gradient, n=1, model="gpt-4o"):
        """ Incorporate feedback gradient into a prompt."""
        transformation_prompt = f"""
        I'm trying to write a multiple choice question answering prompt across different subject areas.
        
        Here is the best performing prompt so far:
        "{self.best_prompt}"
        
        Here is the best prompt's accuracy:
        {self.best_score}

        Here is the CURRENT prompt that needs improvement:
        "{prompt}"

        But it gets the following examples wrong:
        {error_str}

        From a previous analysis, the main problems with the current prompt can be summarized as:
        {feedback_str}

        First, think step by step about how to improve this prompt:
        - Analyze what behaviors the current prompt encourages in the model.
        - Explain how those behaviors lead to the specific errors shown.
        - Decide what instructions should be added, removed, or rephrased so that the model:
        * reasons carefully but does not overcomplicate problems,
        * uses domain knowledge appropriately,
        * double-checks units, magnitudes, and simple numerical checks,
        * and systematically compares the options before choosing an answer.

        After your analysis, design a NEW instruction prompt that:
        - Encourages clear, step-by-step chain-of-thought reasoning on each question.
        - Helps the model judge when detailed calculations are necessary vs when simple reasoning or known facts are enough.
        - Emphasizes selecting the single best answer from the given options (A, B, C, D).
        - Is not the same as the best prompt so far:

        At the very end of your response, output ONLY the final instruction prompt wrapped exactly as:

        <START>
        [final instruction prompt here]
        <END>

        Do not include <START> and <END> anywhere else in your response, except to mark the start and end of the new prompt.

        The new prompt is:
        """
        transformation_prompt = '\n'.join([line.lstrip() for line in transformation_prompt.split('\n')])
        res = utils.chatgpt(transformation_prompt, n=n, model=model)
        new_prompts = []
        with open(self.opt['logs'], 'a') as outf:
            #outf.write(f"error string: {error_string}\n")
            outf.write("---------------------apply gradients---------------\n")
            outf.write(f"prompt: {transformation_prompt}\n")
            outf.write(f"new prompts: {res}\n")
        for r in res:   
            new_prompts += self.parse_tagged_text(r, "<START>", "<END>")
        return new_prompts

    def generate_synonyms(self, prompt_section, n=3):
        """ Generate synonyms for a prompt section."""
        rewriter_prompt = f"Generate a variation of the following instruction while keeping the semantic meaning.\n\nInput: {prompt_section}\n\nOutput:"
        new_instructions = utils.chatgpt(rewriter_prompt, n=n)
        new_instructions = [x for x in new_instructions if x]
        return new_instructions

    def get_gradients(self, prompt, task_section, task, gpt4, texts, choices, labels, preds, responses, model='gpt-4o-mini'):
        """ Get "gradients" for a prompt based on sampled error strings."""
        prompt_feedbacks = []
        for _ in tqdm(range(self.opt['n_gradients']), total=self.opt['n_gradients'], desc='gradients..'):
            error_string = self._sample_error_str(
                texts, choices, labels, preds, responses, task, n=self.opt['errors_per_gradient'])
            gradients = self._get_gradients(
                task_section, error_string, self.opt['gradients_per_error'], n=1, model=model)
            prompt_feedbacks += [(t, error_string) for t in gradients]
        return prompt_feedbacks

    def expand_candidates(self, prompts, task, gpt4, train_exs):
        """ Expand a list of prompts by generating gradient-based successors and 
            synonyms for each section.
        """
        minibatch = random.sample(train_exs, k=self.opt['minibatch_size'])

        new_prompts = []
        for prompt in tqdm(prompts, desc=f'expanding {len(prompts)} prompts'):
            sections = utils.parse_sectioned_prompt(prompt)
            task_section = sections['task'].strip()

            # evaluate prompt on minibatch
            _, texts, labels, preds = task.evaluate(gpt4, prompt, minibatch)

            # get gradients
            new_task_sections = []
            if self.opt['n_gradients'] > 0:
                gradients = self.get_gradients(prompt, task_section, task, gpt4, texts, labels, preds, model=self.opt['gradient_model'])
                new_task_sections = []
                for feedback, error_string in tqdm(gradients, desc='applying gradients'):
                    tmp = self.apply_gradient(
                        task_section, error_string, feedback, self.opt['steps_per_gradient'], model=self.opt['editing_model'])
                    new_task_sections += tmp
            # generate synonyms
            mc_sampled_task_sections = []
            if self.opt['mc_samples_per_step'] > 0:
                for sect in tqdm(new_task_sections + [task_section], desc='mc samples'):
                    mc_sects = self.generate_synonyms(
                        sect, n=self.opt['mc_samples_per_step'])
                    mc_sampled_task_sections += mc_sects

            # combine
            new_sections = new_task_sections + mc_sampled_task_sections
            new_sections = list(set(new_sections)) # dedup
            tmp_new_prompts = [
                prompt.replace(task_section, tmp) 
                for tmp in new_sections
            ]
            
            # filter a little
            if len(new_sections) > self.opt['max_expansion_factor']:
                if self.opt['reject_on_errors']:
                    error_exs = []
                    for i, (t, l, p) in enumerate(zip(texts, labels, preds)):
                        if l != p:
                            error_exs.append({'text': t, 'label': l})
                    error_exs = random.sample(error_exs, min(len(error_exs), 16))

                    # speed up a little
                    tmp_new_prompts = random.sample(tmp_new_prompts, min(len(tmp_new_prompts), self.opt['max_expansion_factor'] * 2))

                    error_scores = self.bf_eval(tmp_new_prompts, error_exs, task, gpt4, self.scorer, max_threads=self.max_threads)
                    tmp_new_prompts = [tmp_new_prompts[i] for i in np.argsort(error_scores)[-self.opt['max_expansion_factor']:]]
                else:
                    tmp_new_prompts = random.sample(tmp_new_prompts, 
                        k=self.opt['max_expansion_factor'])

            new_prompts += tmp_new_prompts

        new_prompts += prompts # add originals
        new_prompts = list(set(new_prompts)) # dedup

        return new_prompts

    def score_candidates(self, prompts, task, gpt4, train_exs):
        """ Score a list of prompts."""
        if len(prompts) == 1:
            return [1.0]

        evals = self.evaluator_fn(
            prompts, train_exs, task, gpt4,
            scorer=self.scorer,
            rounds=self.opt['eval_rounds'],
            num_prompts_per_round=self.opt['eval_prompts_per_round'],
            samples_per_eval=self.opt['samples_per_eval'],
            max_threads=self.max_threads
        )
        return evals


    def iterate_one_prompt(self, prompts, task, gpt4, train_exs):
        """ Expand a list of prompts by generating gradient-based successors and 
            synonyms for each section.
        """
        # minibatch = random.sample(train_exs, k=self.opt['minibatch_size'])
        minibatch = train_exs

        new_prompts = []
        for prompt in tqdm(prompts, desc=f'expanding {len(prompts)} prompts'):
            sections = utils.parse_sectioned_prompt(prompt)
            task_section = sections['task'].strip()
        
            # evaluate prompt on new minibatch
            f1, accuracy, texts, labels, choices, preds, responses = task.evaluate(gpt4, prompt, minibatch, n=self.opt['minibatch_size'])
            if accuracy > self.best_score:
                self.best_score = accuracy
                self.best_prompt = task_section
            #print("texts[0]:", texts[0])
            #print("choices[0]:", choices[0])
            #print("responses[0]:", responses[0])
            #print("labels[0]:", labels[0])
            self.metrics["f1"].append(f1)
            self.metrics['avg_f1'] = sum(self.metrics["f1"]) / len(self.metrics["f1"])

            # get gradients
            new_task_sections = []
            if self.opt['n_gradients'] > 0:
                gradients = self.get_gradients(prompt, task_section, task, gpt4, texts, choices, labels, preds, responses, model=self.opt["gradient_model"])
                # with open(self.opt['out'], 'a') as outf:
                #     outf.write(f"gradients: {gradients}\n")
                for feedback, error_string in tqdm(gradients, desc='applying gradients'):
                    new_task_sections += self.apply_gradient(
                        task_section, error_string, feedback, self.opt['steps_per_gradient'], model=self.opt["editing_model"])
                    
            tmp_new_prompts = [
                prompt.replace(task_section, tmp) 
                for tmp in new_task_sections
            ]
            with open(self.opt['logs'], 'a') as outf:
                #outf.write(f"f1: {self.metrics['f1'][-1]}\n")
                #outf.write(f"overall f1: {self.metrics['avg_f1']}\n")
                outf.write(f"acc: {self.metrics['acc'][-1]}\n")
                outf.write(f"overall acc: {self.metrics['avg_acc']}\n")
            with open(self.opt['out'], 'a') as outf:
                outf.write(f"acc: {self.metrics['acc'][-1]}\n")

            new_prompts += tmp_new_prompts
        return new_prompts