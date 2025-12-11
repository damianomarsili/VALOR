"""Lightweight sandbox executor used for syntax checks in verifier_reward.

It exposes minimal stubs (depth, gd_detect, vqa) and enforces a stable image
path plus presence of a `final_answer` variable in user code.
"""

from __future__ import annotations

import io
import math
import re
import sys
import textwrap
from typing import Any, Tuple

import numpy as np


# API stubs ---------------------------------------------------------------


def depth(img_pth: str, bbox: Any) -> float:
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        raise ValueError("bbox must be a length-4 sequence")
    return 0.5


def gd_detect(img_pth: str, prompt: str):
    toks = [t for t in re.split(r"\s*,\s*", str(prompt).strip()) if t]
    return [{"bbox": [0, 0, 10, 10], "label": tok} for tok in toks]


def vqa(img_pth: str, bbox: Any, prompt: str) -> str:
    return f"dummy answer for {prompt}"


IMG_PATH = ""  # predefined; no reassignment allowed


def run_user(code: str) -> Tuple[float, str]:
    """Execute user code in a constrained namespace.

    Returns (score, logs). score is 1.0 when final_answer is set and img_pth is
    unchanged; otherwise 0.0. Logs capture stdout/stderr and any error notes.
    """

    stdout_buf, stderr_buf = io.StringIO(), io.StringIO()
    old_out, old_err = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = stdout_buf, stderr_buf

    try:
        wrapped = (
            "def __llm_main():\n"
            "    final_answer = None  # default\n"
            "    __exc = None\n"
            "    try:\n" + textwrap.indent(code, "        ") + "\n"
            "    except BaseException as e:\n"
            "        __exc = e\n"
            "    finally:\n"
            "        # If an exception happened, propagate it; otherwise return the last final_answer.\n"
            "        if __exc is not None:\n"
            "            raise __exc\n"
            "        return locals().get('final_answer', None)\n"
            "\n"
            "__llm_result = __llm_main()\n"
            "final_answer = __llm_result\n"
        )

        sandbox = {
            "__name__": "__main__",
            "__builtins__": __builtins__,
            "depth": depth,
            "gd_detect": gd_detect,
            "vqa": vqa,
            "np": np,
            "math": math,
            "img_pth": IMG_PATH,
        }

        before_img = sandbox["img_pth"]
        exec(compile(wrapped, "<solution_code>", "exec"), sandbox)

        pred_answer = sandbox.get("__llm_result")
        if pred_answer is None:
            pred_answer = sandbox.get("final_answer")

        if sandbox.get("img_pth") != before_img:
            print("Error: img_pth was modified")
            return 0.0, stdout_buf.getvalue() + stderr_buf.getvalue()

        if pred_answer is None:
            print("Error: final_answer was not set")
            return 0.0, stdout_buf.getvalue() + stderr_buf.getvalue()

        return 1.0, stdout_buf.getvalue() + stderr_buf.getvalue()

    except Exception as exc:  # keep failure logs but no traceback leak
        print(f"Exception in child: {exc}")
        return 0.0, stdout_buf.getvalue() + stderr_buf.getvalue()

    finally:
        sys.stdout, sys.stderr = old_out, old_err


def mp_entry(code: str, conn) -> None:
    """Multiprocessing entrypoint: sends (score, logs) then closes conn."""

    score, logs = run_user(code)
    conn.send((score, logs))
    conn.close()
