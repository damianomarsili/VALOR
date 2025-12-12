import argparse
import ast
import hashlib
import json
import os
import random
import re
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import torch
from datasets import load_dataset
from groundingdino.util.inference import load_image, load_model, predict
from valor.grounding_training.utils import verify_and_filter_3stage
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

_THINK_RE = re.compile(r"<plan>(.*?)</plan>", re.IGNORECASE | re.DOTALL)
_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.IGNORECASE | re.DOTALL)
_FENCE_RE = re.compile(r"```[ \t]*([A-Za-z0-9_+.\-]*)[^\n]*\n(.*?)\n?```", re.DOTALL)

DEFAULT_DATA_ROOT = Path(os.environ.get("VALOR_DATA_ROOT", ""))
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "data" / "detector_outputs"
DEFAULT_ODVG_DIR = Path(__file__).resolve().parent / "data" / "odvg"
PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "system_prompt.jinja"


@dataclass
class QuestionSample:
    question: str
    image: object
    prefix: str
    image_id: str | None = None


def set_seeds(seed: int) -> None:
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _strip_code_fences(s: str) -> str:
    s = s.strip()
    fence = re.compile(r"^\s*```(?:[a-zA-Z0-9_+-]*)?\s*\n(.*?)\n\s*```\s*$", re.DOTALL)
    m = fence.match(s)
    return m.group(1) if m else s


def parse_llm_response(text: str) -> str:
    """
    Return the code-ish payload from an LLM reply.
    Prefers a single <answer> block, otherwise the first fenced code block.
    """
    answer_matches = list(_ANSWER_RE.finditer(text))
    if len(answer_matches) == 1:
        return _strip_code_fences(answer_matches[0].group(1)).strip()

    fences = list(_FENCE_RE.finditer(text))
    if fences:
        return _strip_code_fences(fences[0].group(2)).strip()

    return text.strip()


def lines_with_gd_detect(src: str) -> List[str]:
    pattern = re.compile(r'gd_detect\s*\([^)]*?([\'"])(.*?)\1')
    return [m.group(2) for m in pattern.finditer(src)]


def load_system_prompt() -> str:
    return PROMPT_PATH.read_text()


def build_text_model(model_name: str):
    processor = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype="auto", device_map="auto"
    )
    return model, processor


def generate_llm_output(model, processor, system_prompt: str, question: str) -> str:
    messages = [{"role": "user", "content": f"{system_prompt}\\nQuestion: {question}"}]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    inputs = processor(text=[text], return_tensors="pt").to("cuda")
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=16384,
            temperature=0.7,
            top_p=0.8,
            top_k=20,
            min_p=0,
        )
    output_ids = generated_ids[0][len(inputs.input_ids[0]) :].tolist()
    return processor.decode(output_ids, skip_special_tokens=True)


def build_grounding_model(config_path: str, weights_path: str):
    model = load_model(config_path, weights_path)
    return model.to("cuda")


def _tensor_to_list(obj: Sequence) -> List:
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    if isinstance(obj, (list, tuple)):
        return [_tensor_to_list(x) for x in obj]
    return obj


def _resolve_image_path(image_field) -> Path | None:
    """
    Support HF Image objects and plain paths.
    """
    if isinstance(image_field, (str, Path)):
        return Path(image_field)
    candidate = getattr(image_field, "filename", None) or getattr(
        image_field, "path", None
    )
    return Path(candidate) if candidate else None


def materialize_image(image_field, target_dir: Path, stem: str) -> Path | None:
    """
    Save/copy the image to target_dir and return the path.
    """
    target_dir.mkdir(parents=True, exist_ok=True)
    if isinstance(image_field, (str, Path)):
        src = Path(image_field)
        if not src.exists():
            return None
        ext = src.suffix or ".jpg"
        dst = target_dir / f"{stem}{ext}"
        try:
            shutil.copy(src, dst)
            return dst
        except Exception:
            return None

    if hasattr(image_field, "save"):
        dst = target_dir / f"{stem}.jpg"
        try:
            image_field.save(dst)
        except Exception:
            try:
                image_field.convert("RGB").save(dst)
            except Exception:
                return None
        return dst

    return None


# --------------------------------------------------------------------------- #
# ODVG helpers (post-processing detector outputs)
# --------------------------------------------------------------------------- #


def norm_label(s: str) -> str:
    return " ".join(str(s).strip().lower().replace("-", " ").split())


def cxcywh_norm_to_xyxy_abs(box, W, H):
    cx, cy, w, h = box
    x1 = int(max(0, min(W - 1, round((cx - w / 2) * W))))
    y1 = int(max(0, min(H - 1, round((cy - h / 2) * H))))
    x2 = int(max(0, min(W - 1, round((cx + w / 2) * W))))
    y2 = int(max(0, min(H - 1, round((cy + h / 2) * H))))
    return [x1, y1, x2, y2]


def record_signature(rec: dict) -> str:
    insts = rec.get("detection", {}).get("instances", [])
    cooked = sorted([(i["category"], tuple(i["bbox"])) for i in insts])
    key = {
        "filename": rec.get("filename"),
        "height": rec.get("height"),
        "width": rec.get("width"),
        "instances": cooked,
    }
    s = json.dumps(key, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def load_existing_signatures(jsonl_path: Path) -> set:
    sigs = set()
    if not jsonl_path.exists():
        return sigs
    with jsonl_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                insts = rec.get("detection", {}).get("instances", [])
                for i in insts:
                    i.setdefault("category", i.get("category", ""))
                    i.setdefault("bbox", i.get("bbox", []))
                sigs.add(record_signature(rec))
            except Exception:
                continue
    return sigs


def load_existing_label_list(out_dir: Path):
    lm, lt = out_dir / "label_map.json", out_dir / "labels.txt"
    if not lm.exists() and not lt.exists():
        return []
    ordered = []
    if lm.exists():
        try:
            m = json.loads(lm.read_text(encoding="utf-8"))
            ordered = [norm_label(v) for _, v in sorted(((int(k), v) for k, v in m.items()))]
        except Exception:
            print("Warning: could not parse label_map.json; ignoring.")
    if lt.exists():
        try:
            t = lt.read_text(encoding="utf-8")
            parsed = ast.literal_eval(t)
            parsed = [norm_label(x) for x in parsed]
            if ordered and parsed != ordered:
                print("Warning: labels.txt disagrees with label_map.json; using label_map.json order.")
            elif not ordered:
                ordered = parsed
        except Exception:
            print("Warning: could not parse labels.txt; ignoring.")
    seen, out = set(), []
    for c in ordered:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def extend_label_list_from_categories(existing_labels, categories, append_order="sorted"):
    exist = set(existing_labels)
    candidates = [c for c in categories if c not in exist]
    new_labels = sorted(set(candidates)) if append_order == "sorted" else list(dict.fromkeys(candidates))
    return existing_labels + new_labels


def inject_label_ids(records, label_list):
    l2i = {c: i for i, c in enumerate(label_list)}
    for r in records:
        for inst in r["detection"]["instances"]:
            inst["label"] = l2i[inst["category"]]


def append_new_records(jsonl_path: Path, recs):
    mode = "a" if jsonl_path.exists() else "w"
    with jsonl_path.open(mode, encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def collect_new_records_from_detector(source_dir: Path):
    """
    Read detector JSON outputs and convert to ODVG-style records.
    Returns dict[dataset_name] -> list[record].
    """
    if not source_dir.exists():
        return {}
    datasets = {}
    for ds in [p for p in source_dir.iterdir() if p.is_dir()]:
        batch_seen = set()
        records = []
        for jf in ds.rglob("*.json"):
            if "artifacts" in jf.parts:
                continue
            try:
                payload = json.loads(jf.read_text(encoding="utf-8"))
            except Exception:
                continue
            boxes = payload.get("post_boxes") or []
            labels = payload.get("post_labels") or []
            img_path = payload.get("image_path")
            if not boxes or not labels or len(boxes) != len(labels) or not img_path:
                continue
            try:
                with Image.open(img_path) as im:
                    W, H = im.size
            except Exception:
                continue

            instances = []
            for b, cat in zip(boxes, labels):
                try:
                    bbox = cxcywh_norm_to_xyxy_abs(b, W, H)
                except Exception:
                    continue
                instances.append({"bbox": bbox, "category": norm_label(cat)})
            if not instances:
                continue
            rec = {
                "filename": img_path,
                "height": H,
                "width": W,
                "detection": {"instances": instances},
            }
            sig = record_signature(rec)
            if sig in batch_seen:
                continue
            batch_seen.add(sig)
            records.append(rec)
        datasets[ds.name] = records
    return datasets


def incremental_merge(detector_dir: Path, out_dir: Path, append_order="sorted"):
    out_dir.mkdir(parents=True, exist_ok=True)
    candidates = collect_new_records_from_detector(detector_dir)

    per_ds_to_add = {}
    all_new_categories = []
    for ds_name, recs in candidates.items():
        jsonl_path = out_dir / f"{ds_name}.odvg.jsonl"
        have = load_existing_signatures(jsonl_path)

        seen_batch = set()
        to_add = []
        for r in recs:
            sig = record_signature(r)
            if sig in have or sig in seen_batch:
                continue
            seen_batch.add(sig)
            to_add.append(r)
            for i in r["detection"]["instances"]:
                all_new_categories.append(i["category"])
        per_ds_to_add[ds_name] = to_add

    existing_labels = load_existing_label_list(out_dir)
    label_list = extend_label_list_from_categories(existing_labels, all_new_categories, append_order=append_order)

    for ds_name, recs in per_ds_to_add.items():
        if not recs:
            print(f"{ds_name}: no new records to append.")
            continue
        inject_label_ids(recs, label_list)
        jsonl_path = out_dir / f"{ds_name}.odvg.jsonl"
        append_new_records(jsonl_path, recs)
        print(f"{ds_name}: appended {len(recs)} records to {jsonl_path}")

    label_map = {str(i): c for i, c in enumerate(label_list)}
    tmp_map = out_dir / "label_map.json.tmp"
    tmp_lab = out_dir / "labels.txt.tmp"
    tmp_map.write_text(json.dumps(label_map, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp_lab.write_text("[" + ", ".join(repr(x) for x in label_list) + "]\n", encoding="utf-8")
    (out_dir / "label_map.json").unlink(missing_ok=True)
    (out_dir / "labels.txt").unlink(missing_ok=True)
    tmp_map.rename(out_dir / "label_map.json")
    tmp_lab.rename(out_dir / "labels.txt")
    print(f"Updated label_map.json and labels.txt with {len(label_list)} classes.")


def _format_removed(pre_boxes, pre_labels, pre_logits, post_boxes) -> Dict[str, list]:
    removed_boxes = []
    removed_labels = []
    removed_logits = []
    for i, box in enumerate(pre_boxes):
        if not any(torch.equal(box, pb) for pb in post_boxes):
            removed_boxes.append(box)
            removed_labels.append(pre_labels[i])
            removed_logits.append(pre_logits[i])
    return {
        "boxes": _tensor_to_list(removed_boxes),
        "labels": removed_labels,
        "logits": _tensor_to_list(removed_logits),
    }


def run_grounding_detection(
    gd_model,
    prompt: str,
    image_path: Path,
    save_root: Path,
) -> Dict[str, object]:
    nouns = [
        noun.strip().replace(" ", "-") for noun in prompt.split(",") if noun.strip()
    ]
    grounded_prompt = " . ".join(nouns)

    _, img_gd = load_image(str(image_path))
    box_threshold = 0.1
    text_threshold = 0.1

    with torch.no_grad():
        boxes, logits, phrases = predict(
            model=gd_model,
            image=img_gd,
            caption=grounded_prompt,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
        )

    obj_list = [part.strip() for part in grounded_prompt.split(".") if part.strip()]
    pre_filtered_boxes, pre_filtered_phrases, pre_filtered_logits = [], [], []
    for i, box in enumerate(boxes):
        phrase = phrases[i].replace(" - ", "-")
        if phrase not in obj_list:
            continue
        pre_filtered_boxes.append(box)
        pre_filtered_phrases.append(phrase)
        pre_filtered_logits.append(logits[i])

    if not pre_filtered_boxes:
        return {
            "prompt": prompt,
            "image_path": str(image_path),
            "pre_boxes": [],
            "pre_labels": [],
            "pre_logits": [],
            "post_boxes": [],
            "post_labels": [],
            "post_logits": [],
            "removed": {"boxes": [], "labels": [], "logits": []},
        }

    run_dir = save_root / f"{image_path.stem}_{time.strftime('%Y%m%d-%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)

    (
        post_filtered_boxes,
        post_filtered_phrases,
        post_filtered_logits,
    ) = verify_and_filter_3stage(
        str(image_path),
        torch.stack(pre_filtered_boxes),
        [p.replace("-", " ") for p in pre_filtered_phrases],
        torch.stack(pre_filtered_logits),
    )

    removed = _format_removed(
        torch.stack(pre_filtered_boxes),
        pre_filtered_phrases,
        torch.stack(pre_filtered_logits),
        post_filtered_boxes,
    )

    return {
        "prompt": prompt,
        "image_path": str(image_path),
        "pre_boxes": _tensor_to_list(torch.stack(pre_filtered_boxes)),
        "pre_labels": [p.replace("-", " ") for p in pre_filtered_phrases],
        "pre_logits": _tensor_to_list(torch.stack(pre_filtered_logits)),
        "post_boxes": _tensor_to_list(post_filtered_boxes),
        "post_labels": [p.replace("-", " ") for p in post_filtered_phrases],
        "post_logits": _tensor_to_list(post_filtered_logits),
        "removed": removed,
    }


def save_record(record: Dict[str, object], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2)


def iter_omni3d_samples(data_root: Path) -> Iterable[QuestionSample]:
    dataset = load_dataset("dmarsili/Omni3D-Bench", split="train")
    dataset = dataset.select(range(min(len(dataset), 400))).with_format("python")
    for sample in dataset:
        img = sample.get("image") or sample.get("image_path")
        question = sample.get("question") or sample.get("text")
        if not img or not question:
            continue
        image_id = sample.get("image_id") or sample.get("image_filename")
        yield QuestionSample(question=question, image=img, prefix="o3db", image_id=image_id)


def iter_tallyqa_samples(data_root: Path) -> Iterable[QuestionSample]:
    dataset = load_dataset("snowclipsed/TallyQA", split="train").with_format("python")
    for sample in dataset:
        img = sample.get("image")
        question = sample.get("question")
        if not img or not question:
            continue
        image_id = sample.get("image_id") or sample.get("image_name") or sample.get("image_filename")
        yield QuestionSample(question=question, image=img, prefix="tallyqa", image_id=image_id)


def iter_gqa_samples(data_root: Path) -> Iterable[QuestionSample]:
    dataset = load_dataset(
        "lmms-lab/GQA", "train_balanced_questions", split="train"
    ).with_format("python")
    for sample in dataset:
        img = sample.get("image")
        question = sample.get("question")
        if not img or not question:
            continue
        image_id = sample.get("imageId") or sample.get("image_id")
        yield QuestionSample(question=question, image=img, prefix="gqa", image_id=image_id)


def iter_vsr_samples(data_root: Path) -> Iterable[QuestionSample]:
    dataset = load_dataset(
        "cambridgeltl/vsr_zeroshot", data_files={"train": "train.jsonl"}
    )["train"].with_format("python")
    images_root = data_root / "vsr" / "images"
    for row in dataset:
        img_path = images_root / row["image"]
        if not img_path.exists():
            continue
        question = f"Is {row['caption']}?"
        yield QuestionSample(question=question, image=img_path, prefix="vsr", image_id=row.get("image"))


DATASET_LOADERS = {
    "omni3d-bench": iter_omni3d_samples,
    "tallyqa": iter_tallyqa_samples,
    "gqa": iter_gqa_samples,
    "vsr": iter_vsr_samples,
}


def generate_dataset(
    name: str,
    samples: Iterable[QuestionSample],
    *,
    model,
    processor,
    system_prompt: str,
    gd_model,
    output_dir: Path,
    start_index: int,
    max_samples: int | None,
) -> None:
    dataset_dir = output_dir / name
    dataset_dir.mkdir(parents=True, exist_ok=True)
    images_dir = dataset_dir / "images"

    for idx, sample in enumerate(tqdm(samples, desc=f"{name}")):
        if idx < start_index:
            continue
        if max_samples is not None and (idx - start_index) >= max_samples:
            break

        try:
            llm_output = generate_llm_output(
                model, processor, system_prompt, sample.question
            )
            image_stem = sample.image_id or f"{sample.prefix}-{idx}"
            local_image_path = materialize_image(sample.image, images_dir, image_stem)
            if not local_image_path:
                continue
            code = parse_llm_response(llm_output)
            if not code:
                continue
            calls = lines_with_gd_detect(code)
            if not calls:
                continue
            for call_idx, call in enumerate(calls):
                record = run_grounding_detection(
                    gd_model=gd_model,
                    prompt=call,
                    image_path=local_image_path,
                    save_root=dataset_dir / "artifacts",
                )
                filename = f"{sample.prefix}-{idx}-{call_idx}.json"
                save_record(record, dataset_dir / filename)
        except Exception as exc:
            print(f"[{name}] failed on index {idx}: {exc}")


def parse_args():
    parser = argparse.ArgumentParser(description="Generate grounding training data.")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["omni3d-bench"],
        choices=DATASET_LOADERS.keys(),
        help="Datasets to process.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help="Base path for local datasets (if needed by loaders).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to write detector outputs.",
    )
    parser.add_argument(
        "--odvg-dir",
        type=Path,
        default=DEFAULT_ODVG_DIR,
        help="Directory to write merged ODVG jsonl/label files.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="Qwen/Qwen3-8B",
        help="LLM for question parsing.",
    )
    parser.add_argument("--seed", type=int, default=2727, help="Random seed.")
    parser.add_argument(
        "--start-index", type=int, default=0, help="Skip samples before this index."
    )
    parser.add_argument(
        "--max-samples", type=int, default=None, help="Max samples per dataset."
    )
    parser.add_argument(
        "--grounding-config",
        type=str,
        default="modules/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
        help="GroundingDINO config path.",
    )
    parser.add_argument(
        "--grounding-weights",
        type=str,
        default="modules/GroundingDINO/weights/groundingdino_swint_ogc.pth",
        help="GroundingDINO weights path.",
    )
    parser.add_argument(
        "--append-order",
        choices=["sorted", "first_seen"],
        default=os.getenv("APPEND_ORDER", "sorted"),
        help="Order for NEW classes added to labels.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seeds(args.seed)

    system_prompt = load_system_prompt()
    model, processor = build_text_model(args.model_name)
    gd_model = build_grounding_model(args.grounding_config, args.grounding_weights)

    for dataset_name in args.datasets:
        loader = DATASET_LOADERS[dataset_name]
        samples = loader(args.data_root)
        generate_dataset(
            dataset_name,
            samples,
            model=model,
            processor=processor,
            system_prompt=system_prompt,
            gd_model=gd_model,
            output_dir=args.output_dir,
            start_index=args.start_index,
            max_samples=args.max_samples,
        )

    incremental_merge(args.output_dir, args.odvg_dir, append_order=args.append_order)


if __name__ == "__main__":
    main()
