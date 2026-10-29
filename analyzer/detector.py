"""YOLOv8 vehicle detector wrapper."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from ultralytics import YOLO


@dataclass(frozen=True)
class VehicleDetection:
    """A single supported vehicle detection."""
    xyxy: np.ndarray
    confidence: float
    class_id: int
    class_name: str


class VehicleDetector:
    """Detect configured road-vehicle classes with YOLOv8."""
    def __init__(self, weights: str, confidence: float, iou: float, vehicle_classes: list[str]) -> None:
        if not 0 <= confidence <= 1 or not 0 <= iou <= 1 or not vehicle_classes:
            raise ValueError("invalid model confidence, IoU, or vehicle class configuration")
        try:
            self.model = YOLO(weights)
        except Exception as error:
            raise RuntimeError(
                f"Could not load YOLO weights '{weights}'. Ensure the path exists or allow the first-run download."
            ) from error
        self.confidence, self.iou, self.vehicle_classes = confidence, iou, set(vehicle_classes)

    def detect(self, frame: np.ndarray) -> list[VehicleDetection]:
        """Run inference and retain configured vehicle categories."""
        result = self.model.predict(frame, conf=self.confidence, iou=self.iou, verbose=False)[0]
        if result.boxes is None or len(result.boxes) == 0:
            return []
        names: Any = result.names
        return [
            VehicleDetection(xyxy, float(score), int(class_id), str(names[class_id]))
            for xyxy, score, class_id in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.conf.cpu().numpy(), result.boxes.cls.cpu().numpy().astype(int))
            if str(names[class_id]) in self.vehicle_classes
        ]
