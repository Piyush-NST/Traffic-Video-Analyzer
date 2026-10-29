"""ByteTrack adapter built on supervision."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import supervision as sv

from .detector import VehicleDetection


@dataclass(frozen=True)
class TrackedVehicle:
    """A vehicle observation with persistent track ID."""
    track_id: int
    xyxy: np.ndarray
    confidence: float
    class_name: str

    @property
    def centroid(self) -> tuple[float, float]:
        return ((float(self.xyxy[0]) + float(self.xyxy[2])) / 2, (float(self.xyxy[1]) + float(self.xyxy[3])) / 2)


class VehicleTracker:
    """Maintain ByteTrack identities across frames."""
    def __init__(self, fps: float, config: dict[str, float | int]) -> None:
        if fps <= 0:
            raise ValueError("video FPS must be positive")
        self.tracker = sv.ByteTrack(track_activation_threshold=float(config["track_activation_threshold"]), lost_track_buffer=int(config["lost_track_buffer"]), minimum_matching_threshold=float(config["minimum_matching_threshold"]), frame_rate=int(round(fps)))

    def update(self, detections: list[VehicleDetection]) -> list[TrackedVehicle]:
        """Associate detections and return confirmed tracks."""
        if detections:
            data = sv.Detections(xyxy=np.asarray([d.xyxy for d in detections], dtype=np.float32), confidence=np.asarray([d.confidence for d in detections], dtype=np.float32), class_id=np.asarray([d.class_id for d in detections]), data={"class_name": np.asarray([d.class_name for d in detections])})
        else:
            data = sv.Detections.empty()
        tracked = self.tracker.update_with_detections(data)
        if tracked.tracker_id is None:
            return []
        names = tracked.data.get("class_name", np.asarray(["vehicle"] * len(tracked)))
        return [TrackedVehicle(int(tid), xyxy, float(conf), str(name)) for xyxy, conf, tid, name in zip(tracked.xyxy, tracked.confidence, tracked.tracker_id, names) if tid is not None]
