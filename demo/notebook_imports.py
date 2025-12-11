"""Shared helpers for the VALOR quickstart notebook."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Tuple
import random
import numpy as np

import matplotlib.pyplot as plt
import torch
from IPython.display import Code, Markdown, display
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# Remove stale eval-only sys.path entries that break package-relative imports.
EVAL_ROOT = PROJECT_ROOT / "eval"
if str(EVAL_ROOT) in sys.path:
    sys.path.remove(str(EVAL_ROOT))
if "eval" in sys.modules and getattr(sys.modules["eval"], "__path__", None) is None:
    # Purge incorrectly loaded top-level module so package imports work.
    del sys.modules["eval"]

from eval.utils import load_system_prompt, parse_llm_response  # type: ignore
from demo.notebook_executor import run_program  # type: ignore

DEFAULT_PROMPT_PATH = PROJECT_ROOT / "valor" / "prompts" / "system_prompt.jinja"


def setup_env() -> None:
    """Set matplotlib and tokenization defaults to match eval harness."""
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


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


def load_image(path: str | Path, size: Tuple[int, int] | None = None) -> Image.Image:
    img = Image.open(path).convert("RGB")
    if size:
        img = img.resize(size)
    return img


def show_image(img: Image.Image, title: str = "Input image") -> None:
    plt.figure(figsize=(6, 6))
    plt.imshow(img)
    plt.axis("off")
    plt.title(title)
    plt.show()


def initialize_specialists():
    """Load gd_detect/vqa/depth models used by generated code (optional)."""
    from specialists import api  # type: ignore

    api.init_models()
    return api


def load_model_and_prompt(
    model_path: str, device: str | torch.device, system_prompt_path: Path | str
):
    print("Loading system prompt...")
    system_prompt = load_system_prompt(system_prompt_path)
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype="auto", device_map="auto"
    )
    print("Finished loading model.")
    return system_prompt, tokenizer, model


def model_fwd(model, tokenizer, system_prompt: str, question: str, device: str) -> str:
    messages = [{"role": "user", "content": f"{system_prompt}\nQuestion: {question}"}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    inputs = tokenizer(text=[text], return_tensors="pt").to(device)
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=16384,
        temperature=0.7,
        top_p=0.8,
        top_k=20,
        min_p=0,
    )
    output_ids = generated_ids[0][len(inputs.input_ids[0]) :]
    output_text = tokenizer.decode(output_ids, skip_special_tokens=True)
    return output_text.replace("exit()", "")


def ask_model(model, tokenizer, system_prompt: str, question: str, device: str):
    raw = model_fwd(model, tokenizer, system_prompt, question, device)
    parsed = parse_llm_response(raw)
    plan, code = parsed.get("plan", ""), parsed.get("code", "")
    return raw, plan, code, parsed


def show_md(text: str) -> None:
    display(Markdown(text))


def show_code_block(code: str, title: str | None = None, language: str = "python"):
    if title:
        show_md(f"**{title}**")
    display(Code(code, language=language))


def execute_with_logging(
    plan_text: str,
    code: str,
    image_path: str | Path,
    question: str,
    ground_truth: str,
    out_dir: str | Path | None = None,
):
    out_dir_path = Path(out_dir) if out_dir else None
    pred_answer, logs, steps = run_program(
        plan_text, code, image_path, question=question, ground_truth=ground_truth
    )
    show_md(f"**Model final answer:** {pred_answer}")
    if ground_truth:
        matches = str(pred_answer).strip().lower() == ground_truth.strip().lower()
        show_md(f"**Ground truth:** {ground_truth}  \nMatch: `{matches}`")
    if out_dir_path:
        log_path = out_dir_path / "run.log"
        if log_path.exists():
            show_md(f"Logs saved to `{log_path}`")

    if steps:
        show_md("### Intermediate results")
        for idx, step in enumerate(steps, start=1):
            stype = step.get("type", "step")
            if stype == "gd_detect":
                show_md(f"Step {idx}: `gd_detect` prompt: `{step.get('prompt','')}`")
                if step.get("image_uri"):
                    show_md(f"![Detections]({step['image_uri']})")
            elif stype == "vqa":
                show_md(
                    f"Step {idx}: `vqa` prompt: `{step.get('prompt','')}`, answer: `{step.get('answer')}`"
                )
            elif stype == "depth":
                show_md(
                    f"Step {idx}: `depth` bbox: `{step.get('bbox')}`, value: `{step.get('depth')}`"
                )
                if step.get("image_uri"):
                    show_md(f"![Depth bbox]({step['image_uri']})")
            elif stype == "assign":
                show_md(
                    f"Step {idx}: `{step.get('var')}` = `{step.get('value')}`"
                )
            else:
                show_md(f"Step {idx}: `{stype}` -> `{step}`")
    return pred_answer


def display_plan(plan_text: str) -> None:
    """Nicely render the model's plan."""
    if not plan_text:
        show_md("[no plan parsed]")
        return
    show_md("### Model plan")
    show_md(f"<pre>{plan_text.strip()}</pre>")


def display_code(code: str) -> None:
    """Nicely render the generated code with syntax highlighting."""
    show_code_block(code, title="Generated code", language="python")
