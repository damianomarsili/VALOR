from groundingdino.util.inference import load_model
from moge.model.v2 import MoGeModel
import cv2
import torch
from openai import OpenAI
import base64
import os
from io import BytesIO
from PIL import Image
from math import floor, ceil

from .utils import get_gd_boxes


GD_MODEL = None
MOGE_MODEL = None
GPT_CLIENT = None

DEPTH_MAP = None


def init_models():
    global GD_MODEL, MOGE_MODEL, GPT_CLIENT
    GD_MODEL = load_model(
        "modules/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
        "modules/GroundingDINO/VALOR-checkpoints/VALOR-GroundingDINO.pth",
    )
    MOGE_MODEL = MoGeModel.from_pretrained("Ruicheng/moge-2-vitl-normal").to("cuda")
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY is not set in the environment.")
    GPT_CLIENT = OpenAI(api_key=key)


def ensure_models_initialized():
    """Lazy-load heavy models on first use."""
    if GD_MODEL is None or MOGE_MODEL is None or GPT_CLIENT is None:
        init_models()


def vqa(img_pth, box=None, prompt="", fmt="xyxy", pad=0.25):
    """
    Answers a query about an object. If `box` is provided, the image is cropped
    to the box (with slight padding) and the crop is sent to the VQA model.
    If `box` is None, the full image is sent.

    Parameters
    ----------
    img_pth : str | pathlib.Path
        Path to the RGB image file to be analysed.
    box : list[int, int, int, int] | None
        Bounding box coordinates. If fmt='xyxy', use [x_min, y_min, x_max, y_max].
        If fmt='xywh', use [x, y, w, h]. If None, use full image.
    prompt : str
        The question to ask the GPT model. Keep questions simple.
    fmt : {"xyxy","xywh"}, optional
        Format of `box`. Default is "xyxy".
    pad : float | int, optional
        Extra padding around the box. If 0<pad<=1, treated as a fraction of the
        box size. If >1, treated as pixels. Default 0.05 (5% of box size).

    Returns
    -------
    str
        The response to the query.
    """
    ensure_models_initialized()

    # Load
    img = Image.open(img_pth).convert("RGB")
    W, H = img.size

    use_full_image = box is None

    if not use_full_image:
        # Parse box
        if fmt == "xyxy":
            x1, y1, x2, y2 = map(int, box)
        elif fmt == "xywh":
            x, y, w, h = map(int, box)
            x1, y1, x2, y2 = x, y, x + w, y + h
        else:
            raise ValueError("fmt must be 'xyxy' or 'xywh'")

        # Normalize & clamp
        if x1 > x2:
            x1, x2 = x2, x1
        if y1 > y2:
            y1, y2 = y2, y1
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(W, x2), min(H, y2)

        # If the box collapses to nothing, fall back to full image
        if x2 <= x1 or y2 <= y1:
            use_full_image = True

    if use_full_image:
        crop = img
        base_prompt = (
            "You will be shown an image. Answer the question about the relevant objects "
            "in the scene. Answer with ONE word or phrase only. If the answer is a number, "
            "respond with the number not the word. Do not include units. Return ratios or "
            "fractions as decimal and do not round. If options are passed you MUST respond with one of them."
        )
    else:
        # Compute padding (fractional -> pixels)
        bw, bh = max(1, x2 - x1), max(1, y2 - y1)
        if pad <= 1.0:
            px, py = pad * bw, pad * bh
        else:
            px = py = float(pad)

        # Expand with padding and clamp to image bounds
        cx1 = max(0, int(floor(x1 - px)))
        cy1 = max(0, int(floor(y1 - py)))
        cx2 = min(W, int(ceil(x2 + px)))
        cy2 = min(H, int(ceil(y2 + py)))

        # Ensure non-empty crop
        if cx2 <= cx1:
            cx2 = min(W, cx1 + 1)
        if cy2 <= cy1:
            cy2 = min(H, cy1 + 1)

        crop = img.crop((cx1, cy1, cx2, cy2))
        base_prompt = (
            "You will be shown a TIGHTLY CROPPED image around the object of interest. "
            "Answer ONLY about the central object in this crop. Answer with ONE word or "
            "phrase only. If the answer is a number, respond with the digit not the word. "
            "Do not include units. Return ratios or fractions as decimals and do not round. If options are "
            "passed you MUST respond with one of them."
        )

    # Encode image (full or cropped) to data URI
    buffer = BytesIO()
    crop.save(buffer, format="PNG")
    b64_image = base64.b64encode(buffer.getvalue()).decode("utf-8")
    image_uri = f"data:image/png;base64,{b64_image}"

    # Call model
    response = GPT_CLIENT.responses.create(
        model="gpt-5-mini",
        reasoning={"effort": "medium"},
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": f"{base_prompt}; {prompt}"},
                    {"type": "input_image", "image_url": image_uri},
                ],
            }
        ],
    )

    out_txt = response.output_text.strip().lower()
    if out_txt and out_txt[-1] in {".", ","}:
        out_txt = out_txt[:-1]
    return out_txt


def gd_detect(img_pth, prompt):
    """
    Run object detection on an image and return the post-processed bounding boxes.
    Not necessarily in the same order as the prompt.

    Parameters
    ----------
    img_pth : str or pathlib.Path
        Path to the RGB image file to be analysed. NOT THE IMAGE – THE PATH.
        DO NOT PASS A CV2 IMAGE.
    prompt  : str
        Natural-language description of the objects to look for. MUST BE A NOUN.
        (e.g., "fireplace, coffee table, sofa"). This string is forwarded unchanged to get_boxes.

    Returns
    -------
    list[dict]:
        A list where each element is a dict {"bbox": [x1, y1, x2, y2], "label": <str>}. The list is NOT necessarily in order of the prompt.
    """
    ensure_models_initialized()

    nouns = [noun.strip().replace(" ", "-") for noun in prompt.split(",")]
    prompt = " . ".join(nouns)
    gd_out = get_gd_boxes(img_pth, prompt, GD_MODEL)
    return gd_out


def depth(img_pth, bbox):
    """
    Estimates the depth of an object specified by a bounding box.

    Parameters
    ----------
    img_pth : str
        Path to the input image file.
    bbox : list[int, int, int, int]
        Object bounding box in form [x1, y1, x2, y2].

    Returns
    -------
    float
        The depth of the object specified by the bounding box.
    """
    ensure_models_initialized()
    global DEPTH_MAP
    if DEPTH_MAP is None:
        DEPTH_MAP = precompute_depth(img_pth)

    x1, y1, x2, y2 = bbox
    c_x = int((x1 + x2) / 2)
    c_y = int((y1 + y2) / 2)

    return DEPTH_MAP[c_y, c_x]


def precompute_depth(img_pth):
    ensure_models_initialized()
    global DEPTH_MAP

    input_img = cv2.cvtColor(cv2.imread(img_pth), cv2.COLOR_BGR2RGB)
    input_img = torch.tensor(
        input_img / 255, dtype=torch.float32, device="cuda"
    ).permute(2, 0, 1)
    output = MOGE_MODEL.infer(input_img)["depth"]
    DEPTH_MAP = output
    return output
