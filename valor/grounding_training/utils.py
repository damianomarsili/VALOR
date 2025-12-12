from __future__ import annotations

import base64
import io
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple, Union

import numpy as np
import torch
from openai import OpenAI
from PIL import Image, ImageDraw, ImageFont


_OPENAI_CLIENT: OpenAI | None = None

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

PASS1_PROMPT = """You are an Object-Detection Verifier.

You will receive:
- CLEAN image (truth)
- ANNOTATED image (same image with numbered boxes + labels). The number corresponds to the box that it is INSIDE OF.
- A numbered list of detections (index → label, [x1,y1,x2,y2])

Your task:
Decide which detections to KEEP and which to DROP.
Return ONLY the JSON required by the schema: {"keep_indices":[...], "drop_indices":[...], "notes":[...]}.

Definitions (judge against the CLEAN image):
- Correct: the dominant object of the box is an instance of the labeled object; label matches; box is reasonably tight (some background or a small part of another object visible in the borders of the box is OK).
- Incorrect → drop for any of:
  • wrong_label — contents don’t match the label or no clear object.
  • bad_box — far too loose/tight, mostly background, contains the entirety of multiple instances, or chops off the object.

Indexing:
- Indices are 1-based and must be fully partitioned into keep or drop. No omissions or repeats.
- keep_indices should be sorted by input order.

Notes (short, per decision):
- "4 wrong_label (label=cat, object=dog)"
- "5 bad_box (cuts off left side)"

Never propose new boxes or relabel. When uncertain, favor precision (drop).
Return ONLY the JSON object—no prose.
"""

BOX_CHECK_PROMPT = """You are a single-box verifier.

Input:
- One CROPPED image.
- One predicted label

Goal:
Decide whether the **main visible object in the crop clearly matches the label**.

Rules (be strict):
- Set keep=true only if the labeled object is the **dominant subject**, is **clearly visible**, and is **recognizably** that class. The correct object should dominate the image.
- Minor background is fine; the object should not be tiny or incidental. It is acceptable if a small part of another object is visible in the borders of the image -- but it should not take up more than 20 percent of the image.

Set keep=false for any of the following:
- Wrong class: the visible object is a different category, or no clear object of the labeled class is present.
- Truncation: the crop cuts off a substantial part so the class cannot be confidently verified.
- Unclear central object: crop includes several objects and it’s not clear which one matches the label.
- Multiple objects: crop includes significant portions/all of several objects of the correct class, not just one.

Output:
Return ONLY JSON matching the caller’s schema, e.g.:
{"keep": true/false, "reason": "very brief justification"}.
No extra text or markdown."""

PASS3_PROMPT = """You are the FINAL-PASS object-detection verifier.

You will receive:
- CLEAN image (truth)
- ANNOTATED image (same image with numbered boxes + labels). The number corresponds to the box that it is INSIDE OF.
- A numbered list of detections (index → label, [x1,y1,x2,y2])

Goal:
Do a last sanity sweep on the remaining detections. Remove any **incorrect** boxes and resolve **residual duplicates**—with special attention to nested (“sub-object”) boxes and boxes that are very close to each other.

What to KEEP (correct):
- The box contains one main, dominant object.
- Label matches the object class.
- Box is reasonably tight without chopping the object.
- It is acceptable if a small part of another object is visible in the borders of the box -- but it should not take up more than 20 percent of the box.

DROP if any of these apply:
- **wrong_label** — contents don’t match the label, or no clear object of that class.
- **duplicate** — another detection refers to the same physical instance (rules below).

Duplicate rules (same/compatible class):
Treat two detections as duplicates of the **same** instance if ANY is true:
1) They substantially overlap and the dominant object of both boxes is the same.
2) One box lies almost entirely inside the other (nested/sub-object case).
3) Their centers and extents are almost the same (nearly coincident), even if overlap isn’t huge.

Resolving duplicates:
- Keep **exactly one** per duplicate group; drop the rest.
- Prefer the box that best covers the full object with the least extra background.
- If the tighter box chops the object, keep the larger.
- If still unsure, keep the earlier index.

Indexing:
- Indices are 1-based and must be fully partitioned into keep or drop. No omissions or repeats.
- keep_indices should follow input order.

Notes (short, per decision):
- "7 duplicate_of 3 (nested; 3 covers full object)"
- "12 duplicate_of 9 (nearly coincident; kept tighter 12)"
- "4 wrong_label (label=fireplace; object=tv)"
- "5 bad_box (too loose; mostly background)"

Output:
Return ONLY the JSON required by the schema: {"keep_indices":[...], "drop_indices":[...], "notes":[...]}.
No prose or markdown."""


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------


def _load_font(px: int) -> ImageFont.FreeTypeFont:
    for name in ["DejaVuSans-Bold.ttf", "Arial.ttf", "Inter-Bold.ttf"]:
        try:
            return ImageFont.truetype(name, px)
        except Exception:
            continue
    return ImageFont.load_default()


def _to_data_url_png(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def _ensure_np(a) -> np.ndarray:
    if torch is not None and isinstance(a, torch.Tensor):
        return a.detach().cpu().numpy()
    return np.asarray(a)


def _slice_like(a, idxs):
    if torch is not None and isinstance(a, torch.Tensor):
        return a[idxs]
    arr = np.asarray(a)
    return arr[idxs]


def new_cxcywh_to_xyxy(
    box: np.ndarray, width: int, height: int
) -> Tuple[int, int, int, int]:
    cx, cy, w, h = box
    x1 = (cx - w / 2.0) * width
    y1 = (cy - h / 2.0) * height
    x2 = (cx + w / 2.0) * width
    y2 = (cy + h / 2.0) * height
    x1, y1, x2, y2 = int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))
    return max(0, x1), max(0, y1), min(width - 1, x2), min(height - 1, y2)


def overlay_detections(
    image: Union[str, Image.Image],
    boxes_cxcywh_norm: Union[np.ndarray, "torch.Tensor"],
    phrases: List[str],
    box_color=(0, 200, 0),
    box_thickness=3,
) -> Tuple[Image.Image, List[Tuple[int, int, int, int]]]:
    img = (
        Image.open(image).convert("RGB")
        if isinstance(image, (str, Path))
        else image.convert("RGB")
    )
    w, h = img.size
    draw = ImageDraw.Draw(img, "RGBA")

    arr = _ensure_np(boxes_cxcywh_norm)
    if arr.ndim != 2 or arr.shape[1] != 4 or len(phrases) != arr.shape[0]:
        raise ValueError("boxes must be (N,4) and phrases length must match")

    base = max(12, int(0.018 * max(w, h)))
    font = _load_font(base)
    small_font = _load_font(int(base * 0.9))

    xyxys: List[Tuple[int, int, int, int]] = []
    for i, (box, label) in enumerate(zip(arr, phrases), start=1):
        x1, y1, x2, y2 = new_cxcywh_to_xyxy(box, w, h)
        xyxys.append((x1, y1, x2, y2))

        for k in range(box_thickness):
            draw.rectangle([x1 - k, y1 - k, x2 + k, y2 + k], outline=box_color)

        badge_r = int(base * 0.9)
        bx = max(x1 + badge_r + 2, badge_r + 2)
        by = max(y1 + badge_r + 2, badge_r + 2)
        draw.ellipse(
            [bx - badge_r, by - badge_r, bx + badge_r, by + badge_r],
            fill=(0, 0, 0, 200),
        )
        num_text = str(i)
        tw, th = draw.textbbox((0, 0), num_text, font=font)[2:]
        draw.text(
            (bx - tw / 2, by - th / 2), num_text, fill=(255, 255, 255, 255), font=font
        )

        label_text = f"{label}"
        ltw, lth = draw.textbbox((0, 0), label_text, font=small_font)[2:]
        lx1, ly1 = max(0, x1), max(0, y1 - lth - 12)
        lx2, ly2 = min(w - 1, lx1 + ltw + 12), min(h - 1, ly1 + lth + 12)
        draw.rectangle([lx1, ly1, lx2, ly2], fill=(0, 0, 0, 180))
        draw.text(
            (lx1 + 6, ly1 + 6), label_text, fill=(255, 255, 255, 255), font=small_font
        )

    return img, xyxys


def _xyxy_from_cxcywh_norm(
    boxes_cxcywh_norm: np.ndarray, im_w: int, im_h: int
) -> np.ndarray:
    cx = boxes_cxcywh_norm[:, 0] * im_w
    cy = boxes_cxcywh_norm[:, 1] * im_h
    w = boxes_cxcywh_norm[:, 2] * im_w
    h = boxes_cxcywh_norm[:, 3] * im_h
    x1 = cx - w / 2.0
    y1 = cy - h / 2.0
    x2 = cx + w / 2.0
    y2 = cy + h / 2.0
    xyxy = np.stack(
        [
            np.clip(x1, 0, im_w - 1),
            np.clip(y1, 0, im_h - 1),
            np.clip(x2, 0, im_w - 1),
            np.clip(y2, 0, im_h - 1),
        ],
        axis=1,
    ).astype(int)
    return xyxy


def _expand_and_clip_box(xyxy, pad_frac: float, im_w: int, im_h: int):
    x1, y1, x2, y2 = xyxy
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    px = int(round(bw * pad_frac))
    py = int(round(bh * pad_frac))
    ex1 = max(0, x1 - px)
    ey1 = max(0, y1 - py)
    ex2 = min(im_w - 1, x2 + px)
    ey2 = min(im_h - 1, y2 + py)
    return (ex1, ey1, ex2, ey2)


def _crop_and_upscale(
    img: Image.Image, box_xyxy, target_long_side: int = 640
) -> Image.Image:
    x1, y1, x2, y2 = box_xyxy
    crop = img.crop((x1, y1, x2, y2))
    w, h = crop.size
    if max(w, h) == 0:
        return crop
    if w >= h:
        new_w = target_long_side
        new_h = max(1, int(round(h * (target_long_side / float(w)))))
    else:
        new_h = target_long_side
        new_w = max(1, int(round(w * (target_long_side / float(h)))))
    return crop.resize((new_w, new_h), resample=Image.BICUBIC)


# ---------------------------------------------------------------------------
# LLM helpers
# ---------------------------------------------------------------------------


def _extract_output_text(resp) -> str:
    txt = getattr(resp, "output_text", None)
    if txt:
        return txt
    chunks = []
    for item in getattr(resp, "output", []) or []:
        for c in getattr(item, "content", []) or []:
            if getattr(c, "type", None) == "output_text":
                chunks.append(getattr(c, "text", ""))
    return "".join(chunks)


def _openai_client() -> OpenAI:
    global _OPENAI_CLIENT
    if _OPENAI_CLIENT is None:
        api_key = os.getenv("OPENAI_API_KEY")
        _OPENAI_CLIENT = OpenAI(api_key=api_key)
    return _OPENAI_CLIENT


def ask_pass_level_filter(
    annotated_image: Image.Image,
    clean_image: Image.Image,
    xyxys: List[Tuple[int, int, int, int]],
    phrases: List[str],
    pass_prompt: str,
    extra_instructions: str = "",
    reasoning_effort: str = "medium",
    model_name: str = "gpt-5-mini",
) -> Dict[str, Any]:
    mapping_lines = []
    for i, (box, label) in enumerate(zip(xyxys, phrases), start=1):
        x1, y1, x2, y2 = box
        mapping_lines.append(f"{i}: label='{label}', box_px=[{x1},{y1},{x2},{y2}]")
    mapping_text = "\n".join(mapping_lines)

    schema_name = "verification_result"
    json_schema_only = {
        "type": "object",
        "properties": {
            "keep_indices": {"type": "array", "items": {"type": "integer"}},
            "drop_indices": {"type": "array", "items": {"type": "integer"}},
            "notes": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["keep_indices", "drop_indices"],
        "additionalProperties": False,
    }

    user_text = (
        "Inputs for verification:\n"
        "• CLEAN_IMAGE below (truth), then ANNOTATED_IMAGE (with boxes+labels).\n"
        "• Detections (1-based index → label, [x1,y1,x2,y2] in px):\n"
        f"{mapping_text}\n\n"
        "Output only JSON with keys keep_indices, drop_indices, notes. "
        "Partition all indices; no omissions."
    )
    if extra_instructions:
        user_text += f"\n\nAdditional context: {extra_instructions}"

    clean_image_data_url = _to_data_url_png(clean_image)
    annotated_image_data_url = _to_data_url_png(annotated_image)

    base_kwargs: Dict[str, Any] = dict(
        model=model_name,
        instructions=(pass_prompt or "").strip(),
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": user_text},
                    {"type": "input_image", "image_url": clean_image_data_url},
                    {"type": "input_image", "image_url": annotated_image_data_url},
                ],
            }
        ],
        reasoning={"effort": reasoning_effort},
    )

    client = _openai_client()
    last_err = None

    # Try schema, then fallback to free-form JSON extraction
    try:
        resp = client.responses.create(
            **base_kwargs,
            response_format={
                "type": "json_schema",
                "json_schema": {"name": schema_name, "schema": json_schema_only},
                "strict": True,
            },
        )
        return json.loads(_extract_output_text(resp))
    except Exception as err:
        last_err = err

    try:
        resp = client.responses.create(
            **base_kwargs,
            text={
                "format": {
                    "type": "json_schema",
                    "name": schema_name,
                    "json_schema": json_schema_only,
                    "strict": True,
                }
            },
        )
        return json.loads(_extract_output_text(resp))
    except Exception as err:
        last_err = err

    try:
        raw_guard = (
            "Return ONLY a JSON object with keys 'keep_indices', 'drop_indices', and 'notes'. "
            "No extra text or markdown."
        )
        resp = client.responses.create(
            **{
                **base_kwargs,
                "instructions": base_kwargs["instructions"] + " " + raw_guard,
            }
        )
        raw = _extract_output_text(resp)
        start, end = raw.find("{"), raw.rfind("}")
        if start != -1 and end > start:
            raw = raw[start : end + 1]
        return json.loads(raw)
    except Exception as err:
        raise RuntimeError(
            f"Pass-level filter failed. Last error: {repr(last_err)} / {repr(err)}"
        )


def ask_box_binary_check(
    crop_image: Image.Image,
    predicted_label: str,
    box_index_1b: int,
    prompt: str,
    reasoning_effort: str = "medium",
    model_name: str = "gpt-5-mini",
) -> Dict[str, Any]:
    schema_name = "binary_check"
    json_schema_only = {
        "type": "object",
        "properties": {
            "keep": {"type": "boolean"},
            "reason": {"type": "string"},
        },
        "required": ["keep"],
        "additionalProperties": False,
    }

    crop_url = _to_data_url_png(crop_image)
    user_text = (
        f"Single detection check.\n"
        f"Index: {box_index_1b}\n"
        f"Predicted label: '{predicted_label}'\n\n"
        f"Decide keep=true if the crop clearly shows the predicted label; otherwise keep=false.\n"
        f"Return ONLY JSON with the given schema."
    )

    base_kwargs: Dict[str, Any] = dict(
        model=model_name,
        instructions=(prompt or "").strip(),
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": user_text},
                    {"type": "input_image", "image_url": crop_url},
                ],
            }
        ],
        reasoning={"effort": reasoning_effort},
    )

    client = _openai_client()
    last_err = None

    try:
        resp = client.responses.create(
            **base_kwargs,
            response_format={
                "type": "json_schema",
                "json_schema": {"name": schema_name, "schema": json_schema_only},
                "strict": True,
            },
        )
        return json.loads(_extract_output_text(resp))
    except Exception as err:
        last_err = err

    try:
        resp = client.responses.create(
            **base_kwargs,
            text={
                "format": {
                    "type": "json_schema",
                    "name": schema_name,
                    "json_schema": json_schema_only,
                    "strict": True,
                }
            },
        )
        return json.loads(_extract_output_text(resp))
    except Exception as err:
        last_err = err

    try:
        guard = (
            "Return ONLY a JSON object with keys 'keep' (boolean) and 'reason' (string). "
            "No extra text or markdown."
        )
        resp = client.responses.create(
            **{**base_kwargs, "instructions": base_kwargs["instructions"] + " " + guard}
        )
        raw = _extract_output_text(resp)
        start, end = raw.find("{"), raw.rfind("}")
        if start != -1 and end > start:
            raw = raw[start : end + 1]
        return json.loads(raw)
    except Exception as err:
        raise RuntimeError(
            f"Binary box check failed. Last error: {repr(last_err)} / {repr(err)}"
        )


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def _to_serializable(x: Any):
    try:
        import numpy as _np
    except Exception:
        _np = None
    try:
        import torch as _torch
    except Exception:
        _torch = None

    if _torch is not None and isinstance(x, _torch.Tensor):
        return x.detach().cpu().tolist()
    if _torch is not None and isinstance(x, (_torch.dtype,)):
        return str(x)

    if _np is not None and isinstance(x, _np.ndarray):
        return x.tolist()
    if _np is not None and isinstance(x, (_np.integer,)):
        return int(x)
    if _np is not None and isinstance(x, (_np.floating,)):
        return float(x)
    if _np is not None and isinstance(x, (_np.bool_,)):
        return bool(x)

    if isinstance(x, (set, tuple)):
        return list(x)
    if hasattr(x, "tolist"):
        try:
            return x.tolist()
        except Exception:
            pass
    return x


def verify_and_filter_3stage(
    image: Union[str, Image.Image],
    boxes_cxcywh_norm: Union[np.ndarray, "torch.Tensor"],
    phrases: List[str],
    logits,
    pass1_prompt: str = PASS1_PROMPT,
    box_check_prompt: str = BOX_CHECK_PROMPT,
    pass3_prompt: str = PASS3_PROMPT,
    crop_pad_frac: float = 0.10,
    crop_upscale_long_side: int = 640,
    reasoning_effort: str = "medium",
    model_name: str = "gpt-5-mini",
) -> Tuple[np.ndarray, List[str], List[Any]]:
    clean_image = (
        Image.open(image).convert("RGB")
        if isinstance(image, (str, Path))
        else image.convert("RGB")
    )
    w, h = clean_image.size

    boxes_norm_np = _ensure_np(boxes_cxcywh_norm)
    n = len(phrases)
    if boxes_norm_np.shape[0] != n:
        raise ValueError("boxes and phrases length mismatch")

    annotated_img_all, xyxys_px_all = overlay_detections(
        clean_image, boxes_norm_np, phrases
    )

    pass1_json = ask_pass_level_filter(
        annotated_image=annotated_img_all,
        clean_image=clean_image,
        xyxys=xyxys_px_all,
        phrases=phrases,
        pass_prompt=pass1_prompt,
        reasoning_effort=reasoning_effort,
        model_name=model_name,
    )

    keep1_1b = [int(i) for i in pass1_json.get("keep_indices", [])]
    keep1_0b = sorted([i - 1 for i in keep1_1b if 1 <= i <= n])

    boxes_after_p1 = _slice_like(boxes_cxcywh_norm, keep1_0b)
    phrases_after_p1 = [phrases[i] for i in keep1_0b]
    logits_after_p1 = [logits[i] for i in keep1_0b]
    xyxy_after_p1 = _xyxy_from_cxcywh_norm(_ensure_np(boxes_after_p1), w, h)

    annotated_img_p1, _ = overlay_detections(
        clean_image, _ensure_np(boxes_after_p1), phrases_after_p1
    )

    pass2_json_by_index: Dict[int, Dict[str, Any]] = {}
    keep2_global_0b: List[int] = []

    for j, glob_idx in enumerate(keep1_0b, start=1):
        idx1b = glob_idx + 1
        xyxy = xyxy_after_p1[j - 1]
        ex1, ey1, ex2, ey2 = _expand_and_clip_box(
            xyxy, pad_frac=crop_pad_frac, im_w=w, im_h=h
        )
        crop_up = _crop_and_upscale(
            clean_image, (ex1, ey1, ex2, ey2), target_long_side=crop_upscale_long_side
        )

        verdict = ask_box_binary_check(
            crop_image=crop_up,
            predicted_label=phrases[glob_idx],
            box_index_1b=idx1b,
            prompt=box_check_prompt,
            reasoning_effort=reasoning_effort,
            model_name=model_name,
        )
        pass2_json_by_index[idx1b] = verdict

        keep_flag = bool(verdict.get("keep", False))
        if keep_flag:
            keep2_global_0b.append(glob_idx)

    keep2_global_0b = sorted(keep2_global_0b)

    boxes_after_p2 = _slice_like(boxes_cxcywh_norm, keep2_global_0b)
    phrases_after_p2 = [phrases[i] for i in keep2_global_0b]
    logits_after_p2 = [logits[i] for i in keep2_global_0b]

    annotated_img_p2, xyxys_px_p2 = overlay_detections(
        clean_image, _ensure_np(boxes_after_p2), phrases_after_p2
    )

    if phrases_after_p2:
        pass3_json = ask_pass_level_filter(
            annotated_image=annotated_img_p2,
            clean_image=clean_image,
            xyxys=xyxys_px_p2,
            phrases=phrases_after_p2,
            pass_prompt=pass3_prompt,
            reasoning_effort=reasoning_effort,
            model_name=model_name,
        )
        keep3_1b = [int(i) for i in pass3_json.get("keep_indices", [])]
        keep3_0b_local = sorted(
            [i - 1 for i in keep3_1b if 1 <= i <= len(phrases_after_p2)]
        )
    else:
        pass3_json = {
            "keep_indices": [],
            "drop_indices": [],
            "notes": ["No survivors after Pass 2."],
        }
        keep3_0b_local = []

    keep_final_global_0b = [keep2_global_0b[i] for i in keep3_0b_local]

    filtered_boxes = _slice_like(boxes_cxcywh_norm, keep_final_global_0b)
    filtered_phrases = [phrases[i] for i in keep_final_global_0b]
    filtered_logits = [logits[i] for i in keep_final_global_0b]

    return (
        filtered_boxes,
        filtered_phrases,
        filtered_logits,
    )
