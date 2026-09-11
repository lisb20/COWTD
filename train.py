"""Train on local annotated imagery. Edit the paths below before running."""
from ultralytics import YOLO

DATA = "data/data.yaml"
MODEL = "models/yolo11n.yaml"  # Local architecture YAML, or a local .pt checkpoint.
OUTPUT = "runs"
NAME = "train"
DEVICE = 0  # GPU index; use "cpu" for CPU.

if __name__ == "__main__":
    model = YOLO(MODEL)
    model.train(data=DATA, epochs=300, imgsz=256, pretrained=True,
                device=DEVICE, project=OUTPUT, name=NAME, exist_ok=False)
