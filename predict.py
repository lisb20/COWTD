"""Infer on local tiles and save class xc yc width height confidence labels."""
from pathlib import Path

from ultralytics import YOLO

MODEL = "models/best.pt"
IMAGES = "data/images/2024/hebei"  # Tile filenames: row_col.png (or .jpg).
OUTPUT = "runs/predict/2024/hebei"
CONFIDENCE = 0.4  # Replace with the threshold selected on the validation set.
DEVICE = 0
BATCH = 16

if __name__ == "__main__":
    output = Path(OUTPUT)
    output.mkdir(parents=True, exist_ok=False)  # Never delete previous results.
    model = YOLO(MODEL)
    seen = set()
    results = model.predict(source=IMAGES, imgsz=256, conf=CONFIDENCE,
                            batch=BATCH, device=DEVICE, stream=True,
                            save=False, save_txt=False)
    for result in results:
        name = Path(result.path).stem
        if name in seen:
            raise ValueError(f"Duplicate tile filename: {name}")
        seen.add(name)
        boxes = result.boxes
        lines = []
        for cls, xywh, score in zip(boxes.cls.cpu().numpy(), boxes.xywhn.cpu().numpy(), boxes.conf.cpu().numpy()):
            lines.append(f"{int(cls)} " + " ".join(f"{float(v):.10g}" for v in (*xywh, score)))
        (output / (name + ".txt")).write_text("\n".join(lines))
