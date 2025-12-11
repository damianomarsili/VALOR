from __future__ import annotations

import re
import numpy as np
import os
from openai import OpenAI

GPT_CLIENT = None


def init_gpt_client():
    global GPT_CLIENT
    key = os.environ.get("OPENAI_API_KEY")
    GPT_CLIENT = OpenAI(api_key=key)


def parse_answer(ans):
    p = r"<answer>(.*?)</answer>"
    match = re.findall(p, ans)
    if len(match) == 0:
        p2 = r"<answer>(.*?)<answer>"
        match = re.findall(p2, ans)
        if len(match) == 0:
            return ans
        return match[0]
    return match[0]


def check_synonyms(question, model_pred, gt):
    if GPT_CLIENT is None:
        init_gpt_client()

    prompt = """
You are responsible for verifying answers. I will give you a predicted answer and the ground truth answer. 
The two may not always be an exact match, your job is to check if they are semantically similar.
You should return a score of 1 if they are semantically roughly the same or 0 if they are not. Please put your answer inside <answer></answer> tags.
Question: {}
Ground Truth: {}
Predicted Answer: {}
"""
    response = GPT_CLIENT.responses.create(
        model="gpt-5-mini",
        input=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": prompt.format(question, gt, model_pred),
                    },
                ],
            }
        ],
    )
    verdict = parse_answer(response.output_text)
    return verdict


def update_execution_accuracies_omni3d_bench(
    acc_metrics, ans_type, pred_answer, gt_answer
):
    mra_thresholds = [0.5, 0.45, 0.40, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05]
    corr = 0
    if ans_type == "int":
        acc_metrics["num_ct_n"] += 1
        try:
            pred_answer = int(pred_answer)
        except Exception:
            return acc_metrics, 0.0
        gt_answer = int(gt_answer)
        if gt_answer == pred_answer:
            acc_metrics["num_ct_correct"] += 1
            corr = 1.0
    elif ans_type == "str":
        if gt_answer in ["yes", "no"]:
            acc_metrics["yn_n"] += 1
            try:
                if gt_answer in pred_answer.lower() or pred_answer.lower() in gt_answer:
                    acc_metrics["yn_correct"] += 1
                    corr = 1.0
            except Exception:
                return acc_metrics, 0.0
        elif gt_answer in ["S", "E", "N", "W"]:
            acc_metrics["multi_n"] += 1
            try:
                if gt_answer == pred_answer[0]:
                    acc_metrics["multi_correct"] += 1
                    corr = 1.0
            except Exception:
                return acc_metrics, 0.0
        else:
            acc_metrics["multi_n"] += 1
            try:
                if gt_answer in pred_answer.lower() or pred_answer.lower() in gt_answer:
                    acc_metrics["multi_correct"] += 1
                    corr = 1.0
            except Exception:
                return acc_metrics, 0.0
    elif ans_type == "float":
        acc_metrics["num_other_n"] += 1
        sample_mra = 0
        for threshold in mra_thresholds:
            try:
                pred_answer = float(pred_answer)
                gt_answer = float(gt_answer)
            except Exception:
                continue
            if abs(gt_answer - pred_answer) / gt_answer < threshold:
                sample_mra += 1
        acc_metrics["num_other_mra"] += sample_mra / len(mra_thresholds)
        corr = sample_mra / len(mra_thresholds)
    return acc_metrics, corr


def update_accuracy(dataset, current_acc, pred_answer, gt_answer, answer_type):
    try:
        correct = 0
        updated_acc = current_acc
        if dataset == "omni3d-bench":
            if isinstance(pred_answer, (bool, np.bool_)):
                pred_answer = "yes" if pred_answer else "no"
            updated_acc, correct = update_execution_accuracies_omni3d_bench(
                current_acc, answer_type, pred_answer, gt_answer
            )
        elif dataset == "robospatial":
            pred = "yes" if (isinstance(pred_answer, bool) and pred_answer) else "no"
            if isinstance(pred_answer, str):
                pred = (
                    "yes"
                    if "yes" in pred_answer.lower() or "true" in pred_answer.lower()
                    else "no"
                )
            if pred == gt_answer:
                correct = 1
                updated_acc = current_acc + 1
        elif dataset == "countbench":
            pred = int(pred_answer)
            if pred == gt_answer:
                updated_acc = current_acc + 1
                correct = 1
        elif dataset == "tallyqa":
            pred_answer = int(pred_answer or 0)
            if pred_answer == int(gt_answer):
                updated_acc = current_acc + 1
                correct = 1
        elif dataset == "vsr":
            if isinstance(pred_answer, bool):
                pred = 1 if pred_answer else 0
            elif isinstance(pred_answer, str):
                pred = (
                    1
                    if "yes" in pred_answer.lower() or "true" in pred_answer.lower()
                    else 0
                )
            else:
                pred = pred_answer
            if pred == gt_answer:
                updated_acc = current_acc + 1
                correct = 1
        elif dataset == "realworldqa":
            pred_answer = str(pred_answer).rstrip(".,")
            gt_answer = str(gt_answer).rstrip(".,")
            if pred_answer.lower() in gt_answer.lower():
                updated_acc = current_acc + 1
                correct = 1
        elif dataset == "blink":
            try:
                pred_answer = str(int(pred_answer)).lower()
            except Exception:
                pred_answer = str(pred_answer).lower()
            if pred_answer == gt_answer:
                updated_acc = current_acc + 1
                correct = 1
        elif dataset == "gqa":
            synonym_output = check_synonyms(
                answer_type, pred_answer, gt_answer
            )  # TODO - update naming. answer_type is actually the question.
            verdict = int(synonym_output)
            if verdict == 1:
                updated_acc = current_acc + 1
                correct = 1

            # # answer_type carries the question text for gqa
            # if str(pred_answer).strip() and str(gt_answer).strip():
            #     # leave semantic check to caller if needed
            #     correct = int(str(pred_answer).strip().lower() in str(gt_answer).strip().lower())
            #     updated_acc += correct

        return updated_acc, correct
    except Exception:
        return current_acc, 0
