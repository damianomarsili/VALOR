from groundingdino.util.inference import load_model, load_image, predict, annotate
import torch


def get_boxes(img_pth, img_gd, prompt, im_height, im_width, img_src=None, model=None):
    BOX_TRESHOLD = 0.35
    TEXT_TRESHOLD = 0.25
    boxes, logits, phrases = predict(
        model=model,
        image=img_gd,
        caption=prompt,
        box_threshold=BOX_TRESHOLD,
        text_threshold=TEXT_TRESHOLD,
    )

    obj_list = [part.strip() for part in prompt.split(".") if part.strip()]

    filtered_boxes, filtered_phrases = [], []
    for i, box in enumerate(boxes):
        phrase = phrases[i].replace(" - ", "-")
        if phrase not in obj_list:
            continue
        filtered_boxes.append(box)
        filtered_phrases.append(phrase)

    if len(filtered_boxes) == 0:
        return []

    filtered_boxes = torch.stack(filtered_boxes)
    obj_list = [part.strip() for part in prompt.split(".") if part.strip()]
    out_list = []
    for i, box in enumerate(boxes):
        box = cxcywh_to_xyxy(box)
        scaled_box = scale_bbox(box, im_height, im_width)
        phrase = phrases[i].replace(" - ", " ")
        for obj in obj_list:
            o_l = obj.lower()
            if phrase == o_l:
                phrase = obj
        out_list.append({"bbox": scaled_box, "label": phrase})

    return out_list


def cxcywh_to_xyxy(box):
    """
    Convert a bounding box from (cx, cy, w, h) to (x_min, y_min, x_max, y_max).
    Input can be a list, tuple, or numpy array of length 4.
    """
    cx, cy, w, h = box
    x_min = cx - w / 2
    y_min = cy - h / 2
    x_max = cx + w / 2
    y_max = cy + h / 2

    return [
        round(x_min.item(), 3),
        round(y_min.item(), 3),
        round(x_max.item(), 3),
        round(y_max.item(), 3),
    ]


def scale_bbox(bbox, height, width):
    xmin_norm, ymin_norm, xmax_norm, ymax_norm = bbox

    x_min = int(xmin_norm * width)
    y_min = int(ymin_norm * height)
    x_max = int(xmax_norm * width)
    y_max = int(ymax_norm * height)

    # Clip to image bounds just in case
    x_min = max(0, min(x_min, width - 1))
    x_max = max(0, min(x_max, width - 1))
    y_min = max(0, min(y_min, height - 1))
    y_max = max(0, min(y_max, height - 1))

    bbox_px = [x_min, y_min, x_max, y_max]
    return bbox_px


def get_gd_boxes(img_pth, prompt, model):
    img_source, img_gd = load_image(img_pth)
    im_height, im_width, _ = img_source.shape
    gd_out = get_boxes(
        img_pth, img_gd, prompt, im_height, im_width, img_src=img_source, model=model
    )

    return gd_out
