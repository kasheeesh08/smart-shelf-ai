
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

MODEL_PATH = Path(__file__).parent / "yolov8n_9class_v3_best.pt"
CONFIDENCE = 0.45
IOU_THRESHOLD = 0.50

DISPLAY_NAMES = {
    0: "Bournvita",
    1: "Mysore Sandal Soap",
    2: "Nivea Body Lotion",
    3: "Nivea Soft Cream",
    4: "Parachute Coconut Oil",
    5: "Patanjali Dant Kanti",
    6: "Society Tea Powder",
    7: "Tresemme Conditioner",
    8: "Tresemme Shampoo",
}

model = YOLO(str(MODEL_PATH))

def detect_products(image, confidence=CONFIDENCE):
    result = model.predict(
        source=image,
        conf=confidence,
        iou=IOU_THRESHOLD,
        verbose=False
    )[0]

    original = Image.fromarray(result.orig_img[:, :, ::-1].copy()).convert("RGB")
    annotated = original.copy()
    draw = ImageDraw.Draw(annotated)
    font = ImageFont.load_default()

    detections = []
    boxes = []

    for box in result.boxes:
        class_id = int(box.cls.item())
        score = float(box.conf.item())
        x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].cpu().tolist()]
        product = DISPLAY_NAMES.get(class_id, model.names[class_id])

        detections.append({
            "product": product,
            "confidence": round(score, 4),
            "bbox_xyxy": [round(x1, 1), round(y1, 1),
                          round(x2, 1), round(y2, 1)]
        })
        boxes.append((x1, y1, x2, y2, product, score, class_id))

    # Draw all boxes first; use a compact label with readable text.
    palette = [
        "#22C55E", "#06B6D4", "#3B82F6", "#A855F7",
        "#F97316", "#EC4899", "#EAB308", "#14B8A6", "#EF4444"
    ]

    for x1, y1, x2, y2, product, score, class_id in boxes:
        color = palette[class_id % len(palette)]
        draw.rectangle((x1, y1, x2, y2), outline=color, width=3)
        label = f"{product} {score:.0%}"
        tb = draw.textbbox((0, 0), label, font=font)
        tw, th = tb[2] - tb[0], tb[3] - tb[1]
        pad = 4

        # Prefer above the box, but place inside if near the image top.
        lx = max(0, min(x1, annotated.width - tw - 2*pad))
        ly = y1 - th - 2*pad
        if ly < 0:
            ly = min(y1 + 2, annotated.height - th - 2*pad)

        draw.rectangle(
            (lx, ly, lx + tw + 2*pad, ly + th + 2*pad),
            fill=color
        )
        draw.text((lx + pad, ly + pad), label, fill="black", font=font)

    counts = dict(Counter(d["product"] for d in detections))
    return annotated, detections, counts
