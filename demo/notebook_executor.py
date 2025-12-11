"""Notebook-friendly executor that returns logs and intermediate steps for inline display."""

from __future__ import annotations

import base64
import io
import math
import os
import sys
import textwrap
from pathlib import Path
from typing import Any, Dict, Tuple
from io import BytesIO

import numpy as np
from PIL import Image, ImageDraw

from eval.specialists import api

ROOT = Path(__file__).resolve().parent.parent


def _draw_boxes(img_pth: str, boxes):
    """Return a base64 PNG with overlaid boxes."""
    img = Image.open(img_pth).convert("RGB")
    draw = ImageDraw.Draw(img)
    for box in boxes:
        x1, y1, x2, y2 = map(int, box["bbox"])
        label = str(box.get("label", ""))
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
        if label:
            draw.text((x1 + 4, y1 + 4), label, fill="red")
    buf = BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")


def _run(code: str, img_pth: str) -> Tuple[Any, str, list]:
    steps = []
    stdout_buf, stderr_buf = io.StringIO(), io.StringIO()
    old_out, old_err = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = stdout_buf, stderr_buf
    old_cwd = os.getcwd()
    abs_img = str(Path(img_pth).expanduser().resolve())
    os.chdir(ROOT)

    def _wrap_gd_detect(path, prompt):
        out = api.gd_detect(path, prompt)
        try:
            img_uri = _draw_boxes(path, out)
        except Exception:
            img_uri = None
        steps.append(
            {"type": "gd_detect", "prompt": prompt, "boxes": out, "image_uri": img_uri}
        )
        return out

    def _wrap_vqa(path, box=None, prompt="", fmt="xyxy", pad=0.25):
        ans = api.vqa(path, box=box, prompt=prompt, fmt=fmt, pad=pad)
        steps.append(
            {"type": "vqa", "prompt": prompt, "box": box, "answer": ans, "fmt": fmt}
        )
        return ans

    def _wrap_depth(path, bbox):
        val = api.depth(path, bbox)
        try:
            img_uri = _draw_boxes(path, [{"bbox": bbox, "label": f"depth={val:.2f}"}])
        except Exception:
            img_uri = None
        steps.append(
            {"type": "depth", "bbox": bbox, "depth": float(val), "image_uri": img_uri}
        )
        return val

    sandbox: Dict[str, Any] = {
        "__name__": "__main__",
        "__builtins__": __builtins__,
        "depth": _wrap_depth,
        "gd_detect": _wrap_gd_detect,
        "vqa": _wrap_vqa,
        "np": np,
        "math": math,
        "img_pth": abs_img,
    }

    try:
        wrapped = (
            "def __llm_main():\n"
            "    final_answer = None\n"
            "    __exc = None\n"
            "    try:\n" + textwrap.indent(code, "        ") + "\n"
            "    except BaseException as e:\n"
            "        __exc = e\n"
            "    finally:\n"
            "        if __exc is not None:\n"
            "            raise __exc\n"
            "        return locals().get('final_answer', None)\n"
            "\n"
            "__llm_result = __llm_main()\n"
        )

        exec(compile(wrapped, "<solution_code>", "exec"), sandbox)
        pred_answer = sandbox.get("__llm_result")
        if pred_answer is None:
            pred_answer = sandbox.get("final_answer")
        if pred_answer is not None:
            steps.append(
                {"type": "assign", "var": "final_answer", "value": pred_answer}
            )
        return pred_answer, stdout_buf.getvalue() + stderr_buf.getvalue(), steps

    except Exception as exc:  # noqa: BLE001
        print(f"Exception in child: {exc}")
        return None, stdout_buf.getvalue() + stderr_buf.getvalue(), steps

    finally:
        sys.stdout, sys.stderr = old_out, old_err
        os.chdir(old_cwd)


def run_program(
    plan_text: str,
    code: str,
    img_pth: str,
    *,
    question: str | None = None,  # kept for signature parity
    ground_truth: str | None = None,  # kept for signature parity
):
    """Execute generated code and return answer, logs, and step data for notebooks."""

    pred_answer, logs, steps = _run(code, img_pth)
    return pred_answer, logs, steps
