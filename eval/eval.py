"""Evaluation harness for reasoning models across multiple vision QA datasets.

Datasets supported:
  omni3d-bench, sam, tallyqa, vsr, gqa, blink, robospatial, countbench, realworldqa.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import tempfile
from pathlib import Path
from typing import List, Tuple
from urllib.parse import urlparse
from urllib.request import urlretrieve

import datasets
import numpy as np
import torch
from tqdm import tqdm

from .executor import run_program
from .metrics import update_accuracy
from .utils import (
    image_to_path_in_dir,
    load_jsonl,
    load_system_prompt,
    parse_llm_response,
)
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------------------------------------------------------------------------
# Environment setup
# ---------------------------------------------------------------------------

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
DEFAULT_PROMPT_PATH = PROJECT_ROOT / "valor" / "prompts" / "system_prompt.jinja"
EVAL_DATA_ROOT = PROJECT_ROOT / "eval" / "data"

# GRIT datasets keep the GRIT-data marker in the path for clarity
GQA_JSON_PATH = EVAL_DATA_ROOT / "GRIT-data" / "gqa" / "gqa_val.jsonl"
GQA_IMG_ROOT = EVAL_DATA_ROOT / "GRIT-data" / "gqa" / "images"
TALLYQA_JSON_PATH = EVAL_DATA_ROOT / "GRIT-data" / "tallyqa" / "tallyqa_val.jsonl"
TALLYQA_IMG_ROOT = EVAL_DATA_ROOT / "GRIT-data" / "tallyqa" / "visual_genome"
VSR_IMG_ROOT = EVAL_DATA_ROOT / "vsr" / "images"
BLINK_IMG_ROOT = EVAL_DATA_ROOT / "blink" / "images"
ROBOSPATIAL_IMG_ROOT = EVAL_DATA_ROOT / "robospatial" / "images"
COUNTBENCH_IMG_ROOT = EVAL_DATA_ROOT / "countbench"
REALWORLDQA_IMG_ROOT = EVAL_DATA_ROOT / "realworldqa"


def seed_everything(seed: int = 42):
    """Set seeds for python, numpy, torch (CPU & CUDA) to ensure reproducibility."""
    
    # Python
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU

    print(f"[seed_everything] Seed set to {seed}")

def execute_and_write_outputs(
    query,
    img_pth,
    plan_text,
    code,
    gt_answer,
    dataset,
    current_acc,
    idx,
    out_dir,
    answer_type="",
):
    out_pth = os.path.join(out_dir, str(idx))
    os.makedirs(out_pth, exist_ok=True)

    pred_answer = run_program(
        plan_text,
        code,
        img_pth,
        question=query,
        ground_truth=gt_answer,
        out_dir=out_pth,
    )

    current_acc, correct = update_accuracy(
        dataset, current_acc, pred_answer, gt_answer, answer_type
    )

    out_fn = os.path.join(out_pth, "output.txt")
    with open(out_fn, "w", encoding="utf-8") as f:
        f.write(f"Question: {query}\n")
        f.write(f"Plan: {plan_text}\n")
        f.write(f"Code out: {code}\n")
        f.write(f"Predicted: {pred_answer}\n")
        f.write(f"GT Answer: {gt_answer}\n")
        f.write(f"Correct: {correct}\n")

    print(f"At idx {idx}, Pred: {pred_answer}, GT: {gt_answer}, Correct: {correct}")

    return current_acc, correct


def _resolve_vsr_image(image_link: str | None) -> Path | None:
    """Download a VSR image link to a temporary location"""

    if not image_link:
        return None

    parsed = urlparse(image_link)
    if parsed.scheme not in {"http", "https"}:
        return None

    suffix = Path(parsed.path).suffix or ".jpg"
    fd, tmp_path = tempfile.mkstemp(prefix="vsr_img_", suffix=suffix)
    os.close(fd)
    urlretrieve(image_link, tmp_path)
    return Path(tmp_path)


# ---------------------------------------------------------------------------
# Model forward
# ---------------------------------------------------------------------------


def model_fwd(model, processor, system_prompt: str, q: str) -> str:
    messages = [{"role": "user", "content": f"{system_prompt}\nQuestion: {q}"}]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    inputs = processor(text=[text], return_tensors="pt").to("cuda")
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=16384,
        temperature=0.7,
        top_p=0.8,
        top_k=20,
        min_p=0,
    )
    output_ids = generated_ids[0][len(inputs.input_ids[0]) :].tolist()
    output_text = processor.decode(output_ids, skip_special_tokens=True)
    return output_text.replace("exit()", "")


# ---------------------------------------------------------------------------
# Dataset evals
# ---------------------------------------------------------------------------


def get_dataset_list(datasets: str) -> List[str]:
    return [d.strip() for d in datasets.split(",") if d.strip()]


def eval_omni3d_bench(model, processor, system_prompt, base_out_dir):
    print("----- Evaluating Omni3D-Bench -----")
    dataset = datasets.load_dataset("dmarsili/Omni3D-Bench", split="train").with_format(
        "python"
    )
    total = len(dataset)
    test_indices = list(range(100))
    questions = dataset.select(test_indices)

    out_dir = os.path.join(base_out_dir, "omni3d-bench")
    tmp_img_dir = Path(out_dir) / "tmp_images"
    tmp_img_dir.mkdir(parents=True, exist_ok=True)

    acc_metrics = {
        "yn_correct": 0,
        "yn_n": 0,
        "num_ct_n": 0,
        "num_ct_correct": 0,
        "multi_correct": 0,
        "multi_n": 0,
        "num_other_n": 0,
        "num_other_mra": 0,
    }

    for i, sample in enumerate(tqdm(questions)):
        raw_img = sample.get("image") or sample.get("image_path")
        img_pth = None
        if isinstance(raw_img, (str, os.PathLike)):
            candidate = Path(raw_img)
            img_pth = str(candidate) if candidate.exists() else None
        if img_pth is None and hasattr(raw_img, "save"):
            img_pth = image_to_path_in_dir(
                raw_img, tmp_img_dir, prefix="omni3d_", suffix=".png"
            )
        if img_pth is None and sample.get("image_filename"):
            candidate = Path(sample["image_filename"])
            img_pth = str(candidate) if candidate.exists() else None
        if not img_pth:
            raise ValueError(f"Could not resolve image path for sample index {i}")

        q = sample.get("question") or sample.get("text")

        parsed = parse_llm_response(model_fwd(model, processor, system_prompt, q))
        plan_text, out_code = parsed["plan"], parsed["code"]

        acc_metrics, _ = execute_and_write_outputs(
            q,
            img_pth,
            plan_text,
            out_code,
            sample.get("answer", ""),
            "omni3d-bench",
            acc_metrics,
            i,
            out_dir,
            sample.get("answer_type", ""),
        )

    total_acc = (
        acc_metrics["yn_correct"]
        + acc_metrics["num_ct_correct"]
        + acc_metrics["multi_correct"]
        + acc_metrics["num_other_mra"]
    ) / (
        acc_metrics["yn_n"]
        + acc_metrics["num_ct_n"]
        + acc_metrics["multi_n"]
        + acc_metrics["num_other_n"]
    )
    print(f"Total Accuracy: {total_acc}")


def eval_gqa(model, processor, system_prompt, base_out_dir):
    print("----- Evaluating GQA -----")
    out_dir = os.path.join(base_out_dir, "gqa")
    questions = load_jsonl(GQA_JSON_PATH)
    score = 0
    for i, q in enumerate(tqdm(questions)):
        img_pth = os.path.join(GQA_IMG_ROOT, q["image"])
        parsed = parse_llm_response(
            model_fwd(model, processor, system_prompt, q["question"])
        )
        plan_text, out_code = parsed["plan"], parsed["code"]
        score, _ = execute_and_write_outputs(
            q["question"],
            img_pth,
            plan_text,
            out_code,
            q["answer"],
            "gqa",
            score,
            i,
            out_dir,
            q["question"],
        )
    print(f"GQA Final Score: {score / len(questions)}")


def eval_tally_qa(model, processor, system_prompt, base_out_dir):
    print("----- Evaluating TallyQA -----")
    questions = load_jsonl(TALLYQA_JSON_PATH)
    out_dir = os.path.join(base_out_dir, "tallyqa")
    score = 0
    for idx, q in enumerate(tqdm(questions)):
        img_pth = os.path.join(TALLYQA_IMG_ROOT, q["image"])
        parsed = parse_llm_response(
            model_fwd(model, processor, system_prompt, q["question"])
        )
        plan_text, out_code = parsed["plan"], parsed["code"]
        score, _ = execute_and_write_outputs(
            q["question"],
            img_pth,
            plan_text,
            out_code,
            q["answer"],
            "tallyqa",
            score,
            idx,
            out_dir,
            "",
        )
    print(f"TallyQA Final Score: {score / len(questions)}")


def eval_vsr(model, processor, system_prompt, base_out_dir):
    print("----- Evaluating VSR -----")
    out_dir = os.path.join(base_out_dir, "vsr")
    dataset = datasets.load_dataset(
        "cambridgeltl/vsr_zeroshot", data_files={"dev": "dev.jsonl"}
    )["dev"].with_format("python")

    score = 0
    for i, q in enumerate(tqdm(dataset)):
        question = f"Is {q['caption']}?"
        resolved = _resolve_vsr_image(q.get("image_link"))
        img_pth = str(resolved)

        parsed = parse_llm_response(
            model_fwd(model, processor, system_prompt, question)
        )
        plan_text, out_code = parsed["plan"], parsed["code"]
        score, _ = execute_and_write_outputs(
            question,
            img_pth,
            plan_text,
            out_code,
            q["label"],
            "vsr",
            score,
            i,
            out_dir,
            "",
        )
    print(f"VSR Score: {score / len(dataset)}")


def blink_transform_query(query: str, gt: str) -> Tuple[str, str]:
    m = re.search(r"[A-Z]", gt.strip().upper())
    if not m:
        raise ValueError("GT must contain a letter option like A/B/C/D.")
    gt_letter = m.group(0)

    query_clean = re.sub(
        r"Respond with only[^.]*\.?", "", query, flags=re.IGNORECASE
    ).strip()
    opt_match = re.search(
        r"Options?\s*(?:are|:)\s*(.*?)(?:\.\s*|$)", query_clean, flags=re.IGNORECASE
    )
    if not opt_match:
        raise ValueError("Could not find options in the query.")
    options_blob = opt_match.group(1).strip()
    question = query_clean[: opt_match.start()].strip().rstrip(".")

    items = [s.strip() for s in options_blob.split(",")]
    label_to_value = {}
    ordered_values = []
    for item in items:
        m_item = re.match(r"^([A-Z])\s*(?::|\)|-)\s*(.+?)\s*$", item)
        if not m_item:
            raise ValueError(f"Unrecognized option format: {item!r}")
        label, value = m_item.group(1), m_item.group(2).strip().rstrip(".")
        label_to_value[label] = value
        ordered_values.append((label, value))

    if gt_letter not in label_to_value:
        raise ValueError(f"GT letter {gt_letter!r} not found in options.")

    values_in_order = [v for _, v in ordered_values]
    normalized_query = f"{question}? Options: {{{','.join(values_in_order)}}}."
    normalized_gt = label_to_value[gt_letter]
    return normalized_query, normalized_gt


def blink_eval_loop(
    dataset,
    n_q,
    model,
    processor,
    system_prompt,
    base_out_dir,
    n_choices,
    dset_name,
):
    out_dir = os.path.join(base_out_dir, dset_name)
    score = 0
    for i in tqdm(range(n_q), desc=dset_name):
        sample = dataset[i]
        img_pth = image_to_path_in_dir(
            sample["image_1"], BLINK_IMG_ROOT
        )
        ans = sample["answer"][1]

        if n_choices == 2:
            query = f"{sample['question']} Options are A: {sample['choices'][0]}, B: {sample['choices'][1]}. Respond with only A or B"
        else:
            query = (
                f"{sample['question']} Options are A: {sample['choices'][0]}, B: {sample['choices'][1]}, "
                f"C: {sample['choices'][2]}, D: {sample['choices'][3]}. Respond with only A, B, C, or D."
            )

        query, ans = blink_transform_query(query, ans)
        parsed = parse_llm_response(model_fwd(model, processor, system_prompt, query))
        plan_text, out_code = parsed["plan"], parsed["code"]

        score, _ = execute_and_write_outputs(
            query, img_pth, plan_text, out_code, ans, "blink", score, i, out_dir, ""
        )

    return score


def eval_blink(model, processor, system_prompt, base_out_dir):
    out_dir = os.path.join(base_out_dir, "blink")
    print("----- Evaluating BLINK -----")
    spat_data = datasets.load_dataset(
        "BLINK-Benchmark/BLINK", "Spatial_Relation", split="val"
    ).with_format("python")
    counting_data = datasets.load_dataset(
        "BLINK-Benchmark/BLINK", "Counting", split="val"
    ).with_format("python")

    print("Evaluating Spatial Relationships")
    spat_score = blink_eval_loop(
        spat_data,
        len(spat_data),
        model,
        processor,
        system_prompt,
        out_dir,
        2,
        "spatial",
    )

    print("Evaluating Counting")
    counting_score = blink_eval_loop(
        counting_data,
        len(counting_data),
        model,
        processor,
        system_prompt,
        out_dir,
        4,
        "counting",
    )

    print("Spatial Score:", spat_score / len(spat_data))
    print("Counting Score:", counting_score / len(counting_data))


def eval_robospatial(model, processor, system_prompt, base_out_dir):
    out_dir = os.path.join(base_out_dir, "robospatial")
    print("----- Evaluating RoboSpatial -----")
    comp_data = datasets.load_dataset(
        "chanhee-luke/RoboSpatial-Home", split="compatibility"
    ).with_format("python")
    conf_data = datasets.load_dataset(
        "chanhee-luke/RoboSpatial-Home", split="configuration"
    ).with_format("python")

    comp_score = 0
    print("Evaluating Compatibility")
    out_dir_compat = os.path.join(out_dir, "compatibility")
    for i, comp_sample in enumerate(tqdm(comp_data)):
        query = comp_sample["question"]
        ans = comp_sample["answer"].lower()
        img_pth = image_to_path_in_dir(
            comp_sample["img"], ROBOSPATIAL_IMG_ROOT
        )
        parsed = parse_llm_response(model_fwd(model, processor, system_prompt, query))
        plan_text, out_code = parsed["plan"], parsed["code"]
        comp_score, _ = execute_and_write_outputs(
            query,
            img_pth,
            plan_text,
            out_code,
            ans,
            "robospatial",
            comp_score,
            i,
            out_dir_compat,
        )

    conf_score = 0
    print("Evaluating Configuration")
    out_dir_conf = os.path.join(out_dir, "configuration")
    for i, conf_sample in enumerate(tqdm(conf_data)):
        query = conf_sample["question"]
        ans = conf_sample["answer"].lower()
        img_pth = image_to_path_in_dir(
            conf_sample["img"], ROBOSPATIAL_IMG_ROOT
        )
        parsed = parse_llm_response(model_fwd(model, processor, system_prompt, query))
        plan_text, out_code = parsed["plan"], parsed["code"]
        conf_score, _ = execute_and_write_outputs(
            query,
            img_pth,
            plan_text,
            out_code,
            ans,
            "robospatial",
            conf_score,
            i,
            out_dir_conf,
        )

    print("Compatibility Score:", comp_score / len(comp_data))
    print("Configuration Score:", conf_score / len(conf_data))


def eval_countbenchqa(model, processor, system_prompt, base_out_dir):
    out_dir = os.path.join(base_out_dir, "countbenchqa")
    print("----- Evaluating CountBenchQA -----")
    countbench_data = datasets.load_dataset("vikhyatk/CountBenchQA")[
        "test"
    ].with_format("python")

    score = 0
    for i, q in enumerate(tqdm(countbench_data)):
        question = q["question"]
        img_pth = image_to_path_in_dir(
            q["image"], COUNTBENCH_IMG_ROOT
        )
        ans = q["number"]
        parsed = parse_llm_response(
            model_fwd(model, processor, system_prompt, question)
        )
        plan_text, out_code = parsed["plan"], parsed["code"]
        score, _ = execute_and_write_outputs(
            question,
            img_pth,
            plan_text,
            out_code,
            ans,
            "countbench",
            score,
            i,
            out_dir,
        )

    print("CountBench Score:", score / len(countbench_data))


def realworld_qa_transform_block_flexible(
    question_block: str, answer_or_gt: str
) -> Tuple[str, str]:
    def clean_tail_punct(s: str) -> str:
        return re.sub(r"[?.!]+\s*$", "", s).strip()

    def norm_text(s: str) -> str:
        s = s.strip().strip("'\"").strip()
        s = re.sub(r"\s+", " ", s)
        s = re.sub(r"[.,;:!?()\[\]{}]", "", s)
        return s.lower().strip()

    gt_raw = answer_or_gt.strip()
    m_hdr = re.match(r"(?is)^\s*(?:answer|gt)\s*:\s*(.*)$", gt_raw, flags=re.I)
    if m_hdr:
        gt_raw = m_hdr.group(1).strip()
    gt_clean = gt_raw.strip().strip("'\"").strip()
    is_single_letter = bool(re.fullmatch(r"[A-Z]", gt_clean.upper()))

    qb_core = re.split(r"(?mi)^\s*Please answer\b", question_block)[0]
    first_opt = re.search(r"(?mi)^\s*([A-Z])\s*(?:[.)]|:|- )\s*(.+)$", qb_core)
    if not first_opt:
        raise ValueError("No options found.")

    question_raw = qb_core[: first_opt.start()].strip()
    question_raw = re.sub(r"(?i)^\s*Query:\s*", "", question_raw).strip()
    had_qmark = "?" in question_raw
    question = clean_tail_punct(question_raw) or question_raw
    question = question + ("?" if had_qmark else ".")

    option_lines = re.findall(r"(?mi)^\s*([A-Z])\s*(?:[.)]|:|-)\s*(.+?)\s*$", qb_core)
    ordered_values = []
    label_to_value = {}
    normalized_value_map = {}
    for label, text in option_lines:
        val = text.strip().rstrip(".")
        lab = label.upper()
        ordered_values.append((lab, val))
        label_to_value[lab] = val
        normalized_value_map[norm_text(val)] = val

    if is_single_letter:
        gt_letter = gt_clean.upper()
        if gt_letter not in label_to_value:
            raise ValueError(f"Answer letter '{gt_letter}' not among options.")
        normalized_gt = label_to_value[gt_letter]
    else:
        key = norm_text(gt_clean)
        if key not in normalized_value_map:
            maybe_letter = re.fullmatch(r'"?([A-Z])"?', gt_clean.strip(), flags=re.I)
            if maybe_letter:
                L = maybe_letter.group(1).upper()
                if L in label_to_value:
                    normalized_gt = label_to_value[L]
                else:
                    raise ValueError("Letter not among options.")
            else:
                raise ValueError(
                    f"Ground-truth text '{gt_clean}' did not match any option."
                )
        else:
            normalized_gt = normalized_value_map[key]

    values_only = [v for _, v in ordered_values]
    normalized_query = f"{question} Options: {{{','.join(values_only)}}}."
    return normalized_query, normalized_gt


def eval_realworldqa(model, processor, system_prompt, base_out_dir):
    out_dir = os.path.join(base_out_dir, "realworldqa")
    print("----- Evaluating RealworldQA -----")
    data = datasets.load_dataset("nirajandhakal/realworldqa")["test"].with_format(
        "python"
    )

    score = 0
    for i, q in enumerate(tqdm(data)):
        question = q["question"]
        ans = q["answer"]
        if "A." in question and "B." in question:
            question, ans = realworld_qa_transform_block_flexible(question, ans)

        img_pth = image_to_path_in_dir(
            q["image"], REALWORLDQA_IMG_ROOT
        )
        parsed = parse_llm_response(
            model_fwd(model, processor, system_prompt, question)
        )
        plan_text, out_code = parsed["plan"], parsed["code"]
        score, _ = execute_and_write_outputs(
            question,
            img_pth,
            plan_text,
            out_code,
            ans,
            "realworldqa",
            score,
            i,
            out_dir,
            "",
        )

    print("RealworldQA Accuracy:", score / len(data))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(args):
    dataset_names = get_dataset_list(args.datasets)
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    seed_everything(42)

    system_prompt = load_system_prompt(Path(args.system_prompt))

    processor = AutoTokenizer.from_pretrained("glab-caltech/VALOR-8B")
    model = AutoModelForCausalLM.from_pretrained(
        "glab-caltech/VALOR-8B", torch_dtype="auto", device_map="auto"
    )

    with torch.inference_mode():
        if "omni3d-bench" in dataset_names:
            eval_omni3d_bench(
                model,
                processor,
                system_prompt,
                out_dir,
            )
        if "tallyqa" in dataset_names:
            eval_tally_qa(
                model,
                processor,
                system_prompt,
                out_dir,
            )
        if "vsr" in dataset_names:
            eval_vsr(
                model,
                processor,
                system_prompt,
                out_dir,
            )
        if "gqa" in dataset_names:
            eval_gqa(
                model,
                processor,
                system_prompt,
                out_dir,
            )
        if "blink" in dataset_names:
            eval_blink(
                model,
                processor,
                system_prompt,
                out_dir,
            )
        if "robospatial" in dataset_names:
            eval_robospatial(
                model,
                processor,
                system_prompt,
                out_dir,
            )
        if "countbench" in dataset_names:
            eval_countbenchqa(
                model,
                processor,
                system_prompt,
                out_dir,
            )
        if "realworldqa" in dataset_names:
            eval_realworldqa(
                model,
                processor,
                system_prompt,
                out_dir,
            )


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate reasoning QA models")
    parser.add_argument(
        "--datasets",
        type=str,
        default="omni3d-bench",
        help="Comma-separated list of datasets to eval. Available: [omni3d-bench, gqa, tallyqa, vsr, blink, robospatial, countbench, realworldqa, sam]",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="eval_outputs",
        help="Output directory for eval outputs.",
    )
    parser.add_argument(
        "--system_prompt",
        type=str,
        default=str(DEFAULT_PROMPT_PATH),
        help="Path to the system prompt template.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    try:
        import multiprocessing as mp

        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    main(parse_args())
