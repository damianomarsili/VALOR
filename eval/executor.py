"""Sandbox executor that uses the real API functions."""

from __future__ import annotations

import io
import math
import os
import sys
import textwrap
from typing import Any, Dict, Tuple

import numpy as np

from .specialists import api


def _run(code: str, img_pth: str) -> Tuple[Any, str]:
    """Execute user code, returning (final_answer, combined_logs)."""

    stdout_buf, stderr_buf = io.StringIO(), io.StringIO()
    old_out, old_err = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = stdout_buf, stderr_buf

    sandbox: Dict[str, Any] = {
        "__name__": "__main__",
        "__builtins__": __builtins__,
        "depth": api.depth,
        "gd_detect": api.gd_detect,
        "vqa": api.vqa,
        "np": np,
        "math": math,
        "img_pth": img_pth,
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
        return pred_answer, stdout_buf.getvalue() + stderr_buf.getvalue()

    except Exception as exc:
        print(f"Exception in child: {exc}")
        return None, stdout_buf.getvalue() + stderr_buf.getvalue()

    finally:
        sys.stdout, sys.stderr = old_out, old_err


def run_program(
    plan_text: str,
    code: str,
    img_pth: str,
    *,
    question: str | None = None,
    ground_truth: str | None = None,
    out_dir: str | os.PathLike | None = None,
) -> Any:
    """Execute generated code and optionally persist logs."""

    pred_answer, logs = _run(code, img_pth)

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "run.log"), "w", encoding="utf-8") as f:
            f.write(f"Question: {question}\n")
            f.write(f"Ground truth: {ground_truth}\n")
            f.write(f"Plan:\n{plan_text}\n\n")
            f.write("Logs:\n")
            f.write(logs)
        with open(os.path.join(out_dir, "code.py"), "w", encoding="utf-8") as f:
            f.write(code)

    return pred_answer
