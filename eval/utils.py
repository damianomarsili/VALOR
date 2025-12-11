from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path
from typing import Dict, List

from PIL import Image

_THINK_RE = re.compile(r"<plan>(.*?)</plan>", re.IGNORECASE | re.DOTALL)
_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.IGNORECASE | re.DOTALL)
_FENCE_RE = re.compile(r"```[ \t]*([A-Za-z0-9_+.\-]*)[^\n]*\n(.*?)\n?```", re.DOTALL)


def _strip_code_fences(s: str) -> str:
    s = s.strip()
    fence = re.compile(r"^\s*```(?:[a-zA-Z0-9_+-]*)?\s*\n(.*?)\n\s*```\s*$", re.DOTALL)
    m = fence.match(s)
    return m.group(1) if m else s


def parse_llm_response(text: str) -> Dict[str, object]:
    """Extract plan and code from LLM output with a simple score flag."""

    think_matches = list(_THINK_RE.finditer(text))
    answer_matches = list(_ANSWER_RE.finditer(text))

    plan = think_matches[0].group(1).strip() if len(think_matches) == 1 else ""

    fences = list(_FENCE_RE.finditer(text))
    raw_code = ""
    if len(answer_matches) == 1:
        raw_code = answer_matches[0].group(1)
    elif len(fences) == 1:
        raw_code = fences[0].group(2)
    code = _strip_code_fences(raw_code).strip() if raw_code else text.strip()

    score = 1.0 if plan and code else 0.0
    return {"plan": plan, "code": code, "score": score}


def parse_answer(ans: str) -> str:
    m = re.search(r"<answer>(.*?)</answer>", ans, flags=re.DOTALL)
    return m.group(1) if m else ans


def load_system_prompt(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8")


def load_jsonl(path: str | Path) -> List[dict]:
    out: List[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def image_to_path_in_dir(
    img: Image.Image,
    out_dir: str | os.PathLike,
    *,
    format: str | None = None,
    prefix: str = "tmp_",
    suffix: str | None = None,
    **save_kwargs,
) -> str:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fmt = (format or getattr(img, "format", None) or "JPEG").upper()
    ext = suffix or (".jpg" if fmt == "JPEG" else f".{fmt.lower()}")

    fd, path = tempfile.mkstemp(prefix=prefix, suffix=ext, dir=str(out_dir))
    os.close(fd)
    try:
        if fmt == "JPEG" and img.mode not in ("L", "RGB", "CMYK", "YCbCr"):
            img = img.convert("RGB")
        img.save(path, format=fmt, **save_kwargs)
        return path
    except Exception:
        try:
            os.remove(path)
        except OSError:
            pass
        raise
