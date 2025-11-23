import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.trainer import DualSupervisionTrainer

if __name__ == "__main__":

    TEACHER_WEIGHTS = "path/to/teacher.pt"
    DATA_YAML = "data/data.yaml"

    trainer = DualSupervisionTrainer(
        teacher_model_path=TEACHER_WEIGHTS,
        overrides={
            "model": "yolov8s-seg.pt",
            "data": DATA_YAML,
            "epochs": 50,
            "imgsz": 640,
            "project": "runs/train",
            "name": "cabbage_dual_supervision"
        }
    )
    
    trainer.train()