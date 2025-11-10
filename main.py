import requests
import os
import evaluators
import concurrent.futures
from tqdm import tqdm
import time
import json
import argparse
import scorers
import tasks
import predictors
import optimizers
import math
import random

subjects = [
    "abstract_algebra",
    "anatomy",
    "astronomy",
    "business_ethics",
    "clinical_knowledge",
    "college_biology",
    "college_chemistry",
    "college_computer_science",
    "college_mathematics",
    "college_medicine",
    "college_physics",
    "computer_security",
    "conceptual_physics",
    "econometrics",
    "electrical_engineering",
    "elementary_mathematics",
    "formal_logic",
    "global_facts",
    "high_school_biology",
    "high_school_chemistry",
    "high_school_computer_science",
    "high_school_european_history",
    "high_school_geography",
    "high_school_government_and_politics",
    "high_school_macroeconomics",
    "high_school_mathematics",
    "high_school_microeconomics",
    "high_school_physics",
    "high_school_psychology",
    "high_school_statistics",
    "high_school_us_history",
    "high_school_world_history",
    "human_aging",
    "human_sexuality",
    "international_law",
    "jurisprudence",
    "logical_fallacies",
    "machine_learning",
    "management",
    "marketing",
    "medical_genetics",
    "moral_disputes",
    "moral_scenarios",
    "nutrition",
    "philosophy",
    "prehistory",
    "professional_accounting",
    "professional_law",
    "professional_medicine",
    "professional_psychology",
    "public_relations",
    "security_studies",
    "sociology",
    "us_foreign_policy",
    "virology",
    "world_religions",
]

def get_task_class(task_name):
    if task_name == 'ethos':
        return tasks.EthosBinaryTask
    elif task_name == 'jailbreak':
        return tasks.JailbreakBinaryTask
    elif task_name == 'liar':
        return tasks.DefaultHFBinaryTask
    elif task_name == 'ar_sarcasm':
        return tasks.DefaultHFBinaryTask
    elif task_name == 'mmlu':
        return tasks.MMLUTask
    else:
        raise Exception(f'Unsupported task: {task_name}')


def get_evaluator(evaluator):
    if evaluator == 'bf':
        return evaluators.BruteForceEvaluator
    elif evaluator in {'ucb', 'ucb-e'}:
        return evaluators.UCBBanditEvaluator
    elif evaluator in {'sr', 's-sr'}:
        return evaluators.SuccessiveRejectsEvaluator
    elif evaluator == 'sh':
        return evaluators.SuccessiveHalvingEvaluator
    else:
        raise Exception(f'Unsupported evaluator: {evaluator}')

def get_scorer(scorer):
    if scorer == '01':
        return scorers.Cached01Scorer
    elif scorer == 'll':
        return scorers.CachedLogLikelihoodScorer
    else:
        raise Exception(f'Unsupported scorer: {scorer}')

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', default='mmlu')
    parser.add_argument('--data_dir', default='data/mmlu')
    parser.add_argument('--prompts', default='prompts/mmlu.md')
    parser.add_argument('--task_model', default='gpt-4o-mini')
    parser.add_argument('--gradient_model', default='gpt-4o-mini')
    parser.add_argument('--editing_model', default='gpt-4o-mini')
    # parser.add_argument('--config', default='default.json')
    parser.add_argument('--out', default='expts/mmlu_test0.txt')
    parser.add_argument('--max_threads', default=32, type=int)
    parser.add_argument('--temperature', default=0.0, type=float)

    parser.add_argument('--optimizer', default='nl-gradient')
    parser.add_argument('--rounds', default=6, type=int)

    parser.add_argument('--minibatch_size', default=64, type=int)
    parser.add_argument('--n_gradients', default=1, type=int)
    parser.add_argument('--errors_per_gradient', default=4, type=int)
    parser.add_argument('--gradients_per_error', default=1, type=int)


    parser.add_argument('--steps_per_gradient', default=1, type=int)
    parser.add_argument('--mc_samples_per_step', default=0, type=int)
    parser.add_argument('--max_expansion_factor', default=8, type=int)
    parser.add_argument('--beam_size', default=4, type=int)
    parser.add_argument('--n_test_exs', default=100, type=int)
    parser.add_argument('--engine', default="chatgpt", type=str)

    parser.add_argument('--evaluator', default="bf", type=str)
    parser.add_argument('--scorer', default="01", type=str)
    parser.add_argument('--eval_rounds', default=2, type=int)
    parser.add_argument('--eval_prompts_per_round', default=2, type=int)
    # calculated by s-sr and sr
    parser.add_argument('--samples_per_eval', default=32, type=int)
    parser.add_argument('--c', default=1.0, type=float, help='exploration param for UCB. higher = more exploration')
    parser.add_argument('--knn_k', default=2, type=int)
    parser.add_argument('--knn_t', default=0.993, type=float)
    parser.add_argument('--reject_on_errors', action='store_true') 
    
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = get_args()

    config = vars(args)

    config['eval_budget'] = config['samples_per_eval'] * config['eval_rounds'] * config['eval_prompts_per_round']
    
    task = get_task_class(args.task)(args.data_dir, args.max_threads)
    scorer = get_scorer(args.scorer)()
    evaluator = get_evaluator(args.evaluator)(config)
    bf_eval = get_evaluator('bf')(config)
    predictor = predictors.MMLUPredictor(config)

    optimizer = optimizers.OnlineProTeGi(
        config, evaluator, scorer, args.max_threads, bf_eval)

    if os.path.exists(args.out):
        os.remove(args.out)

    print(config)

    with open(args.out, 'a') as outf:
        outf.write(json.dumps(config) + '\n')

    candidates = [open(fp.strip()).read() for fp in args.prompts.split(',')]

    SEED = 42
    random.seed(SEED)
    random.shuffle(subjects)

    # loop for each subject
    for i, subject in enumerate(subjects, start=1):
        print(f"==========STARTING SUBJECT {i}: {subject}==========")
        with open(args.out, 'a') as outf:
            outf.write(f"==========STARTING SUBJECT {i}: {subject}==========\n")
        task.subject_dir = f'{task.data_dir}/{subject}'
        train_exs = task.get_train_examples()
        test_exs = task.get_test_examples()
        
        for round in tqdm(range(1, config['rounds'] + 1)):
            print("STARTING ROUND ", round)
            start = time.time()

            # expand candidates
            with open(args.out, 'a') as outf:
                outf.write(f"======== ROUND {round} =========\n")
                outf.write(f'current prompt: {candidates}\n')
            new_prompts = optimizer.iterate_one_prompt(candidates, task, predictor, train_exs)
            if new_prompts:
                candidates = new_prompts
            else:
                with open(args.out, 'a') as outf:
                    outf.write(f"iterate failed, continuing with current prompt\n")
        
        # score candidates
        scores = optimizer.score_candidates(candidates, task, predictor, train_exs)
        #[scores, candidates] = list(zip(*sorted(list(zip(scores, candidates)), reverse=True)))

        # record candidates, estimated scores, and true scores
        with open(args.out, 'a') as outf:
            # outf.write(f'{time.time() - start}\n')
            outf.write(f'{scores}\n')
        
        # metrics = []
        # for candidate, score in zip(candidates, scores):
        #     f1, texts, labels, preds = task.evaluate(gpt4, candidate, test_exs, n=args.n_test_exs)
        #     metrics.append(f1)
        # with open(args.out, 'a') as outf:  
        #     outf.write(f'test set accuracy: {metrics}\n')

    print("DONE!")
