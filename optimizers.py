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
        res = utils.chatgpt(gradient_prompt, n=n, temperature = 1)
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
        res = utils.chatgpt(transformation_prompt, n=n, temperature=1)
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
        I'm trying to write a multiple-choice question answering prompt.

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

        You will diagnose why the CURRENT prompt fails, using evidence from the model’s responses.

        IMPORTANT OUTPUT GOAL:
        Your final output should be a small set of missing/ambiguous DECISION RULES the CURRENT prompt fails to enforce.
        Do NOT write a new prompt. Do NOT rewrite instructions. Only diagnose what is missing/unclear.

        For each example, do ALL of the following:
        1. Restate the question in your own words, including what quantity/concept must be determined and what constraints/assumptions are given.
        2. Identify the correct option (letter) and give a justification based on the key principle/relationship.
        3. Locate the model’s failure point. Quote or precisely paraphrase the first step/claim where the model’s reasoning diverges from the correct approach. Name the type of error.
        4. Explain why the chosen answer is wrong or incomplete. State what the model concluded vs what should be concluded. State the minimal correction needed.
        5. Prompt-to-error link (be specific). Cite the exact phrase(s) in the CURRENT PROMPT that likely encouraged the failure mode. Explain the causal chain as: Prompt phrase → encouraged behavior → observed failure. Avoid generic statements like “not enough emphasis” unless you specify which check or decision rule is missing.

        After analyzing all examples, synthesize the findings into a diagnosis of the CURRENT PROMPT’s weaknesses:
        Provide 6-7 core behavioral/policy failures (e.g., “allows choosing the closest option instead of re-checking when results don’t match”).
        For each weakness, reference at least two examples that demonstrate it.
        Phrase weaknesses as missing or ambiguous decision rules that the prompt currently fails to enforce.
        Do NOT propose a new prompt or give rewritten instructions—only diagnose what is missing/unclear.

        Wrap ONLY this final synthesized description with <START> and <END>.
        Do not include <START> or <END> anywhere else in your response.
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
        I'm trying to write a multiple choice question answering prompt.
        
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

        Task: Write a NEW instruction prompt that improves the CURRENT prompt by directly fixing the weaknesses above.

        Before writing the final prompt, produce a structured analysis (bullets) that:
        1) Lists the top failure modes.
        2) For each failure mode, propose concrete, atomic instruction rules that prevent it. Each rule must be written in one of these formats:
        - "IF ... THEN ...; VERIFY ..."
        - "ALWAYS ...; VERIFY ..."
        - "NEVER ...; INSTEAD ...; VERIFY ..."
        The VERIFY clause must be falsifiable.
        3) Decide what to remove or de-emphasize from the CURRENT prompt to reduce hallucination, unnecessary verbosity, or shallow “pattern matching”.


        Then produce the NEW instruction prompt with these constraints:
        - Must be at most {self.opt['max_prompt_length']} tokens.
        - Must be clearly structured with bullets or numbered steps.
        - Must explicitly include the decision rules identified in the analysis.
        - Must not reuse sentences from the best-performing prompt.
        - Do not specify the output format; that is handled elsewhere.

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
            extracted = self.parse_tagged_text(r, "<START>", "<END>")
            for p in extracted:
                p = p.strip()

                p = self._compress_prompt_to_budget(
                    p,
                    target_tokens=self.opt.get("max_prompt_length", 1000),
                    hard_max_tokens=int(self.opt.get("max_prompt_length", 1000) * 1.2)
                )

                new_prompts.append(p)
        return new_prompts

    def _compress_prompt_to_budget(
        self,
        prompt_text: str,
        target_tokens: int = 1000,
        hard_max_tokens: int = 1200,
        model: str = "gpt-4o-mini",
        n: int = 1,
        max_passes: int = 6,
    ) -> str:
        """
        Lossless-ish compression: rewrite to fit within target_tokens.
        Only triggers if prompt exceeds hard_max_tokens (soft limit behavior).
        """
        tok = utils._count_tokens(prompt_text, model=model)
        if tok <= hard_max_tokens:
            return prompt_text  # soft limit: allow small overruns
        print(f"\nPrompt too long at {tok} tokens\n")
        compressed = prompt_text
        for i in range(max_passes):
            compressor_prompt = f"""
            You are a prompt compressor.

            The current prompt has {tok} tokens after {i} compression passes. There are {max_passes - i} passes remaining.
            ABSOLUTE REQUIREMENT: Your output MUST be <= {target_tokens} tokens.

            Compression policy:
            - Preserve core decision rules and constraints.
            - Remove redundancy and merge overlapping rules aggressively.
            - If still too long, DELETE lowest-priority guidance first (style tips, hedging, extra explanation).
            - Remove examples and meta-commentary entirely.
            - Output ONLY the rewritten prompt text (no commentary, no tags, no token counts).

            INPUT PROMPT:
            {compressed}
            """.strip()

            res = utils.chatgpt(compressor_prompt, n=n, model=model, temperature=0.3)
            candidate = res[0] if isinstance(res, list) and len(res) else (res or "")
            candidate = candidate.strip() if candidate else ""
            newtok = utils._count_tokens(candidate, model=model)
            if(newtok < tok):
                tok = newtok
                compressed = candidate
            if tok <= hard_max_tokens:
                break
            print(f"Still too long at {tok} tokens after compression pass\n")
        if tok > hard_max_tokens:
            with open(self.opt['out'], 'a') as outf:
                outf.write(f"Warning: prompt still too long at {tok} tokens after compression\n")
            compressed = utils.hard_truncate(compressed, hard_max_tokens, model=model)
        with open(self.opt['out'], 'a') as outf:
            outf.write(f"Final prompt at {tok} tokens\n")
        return compressed

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
            self.metrics["f1"].append(f1)
            self.metrics['avg_f1'] = sum(self.metrics["f1"]) / len(self.metrics["f1"])

            # get gradients
            new_task_sections = []
            if self.opt['n_gradients'] > 0:
                gradients = self.get_gradients(prompt, task_section, task, gpt4, texts, choices, labels, preds, responses, model=self.opt["gradient_model"])
                # with open(self.opt['out'], 'a') as outf:
                #     outf.write(f"gradients: {gradients}\n")
                for feedback, error_string in tqdm(gradients[: self.opt["n_gradients"]], desc='applying gradients'):
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