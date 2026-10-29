"""Calibrated, smoothed speed estimation."""

from __future__ import annotations

from collections import deque


class SpeedEstimator:
    """Estimate per-track km/h from centroid displacement."""
    def __init__(self, fps: float, pixels_per_meter: float, speed_smoothing_frames: int, max_speed_kmh: float) -> None:
        if fps <= 0 or pixels_per_meter <= 0 or speed_smoothing_frames < 1 or max_speed_kmh <= 0:
            raise ValueError("invalid calibration values")
        self.fps, self.pixels_per_meter, self.window, self.maximum = fps, pixels_per_meter, speed_smoothing_frames, max_speed_kmh
        self.previous: dict[int, tuple[float, float]] = {}
        self.history: dict[int, deque[float]] = {}

    def update(self, track_id: int, centroid: tuple[float, float]) -> float:
        """Store a position and return smoothed speed in km/h."""
        previous = self.previous.get(track_id)
        self.previous[track_id] = centroid
        values = self.history.setdefault(track_id, deque(maxlen=self.window))
        if previous is None:
            return 0.0
        distance_px = ((centroid[0] - previous[0]) ** 2 + (centroid[1] - previous[1]) ** 2) ** 0.5
        values.append(min((distance_px / self.pixels_per_meter) * self.fps * 3.6, self.maximum))
        return sum(values) / len(values)
