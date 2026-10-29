"""Polygon-occupancy congestion checks."""

from __future__ import annotations

import cv2
import numpy as np

from .tracker import TrackedVehicle


class CongestionDetector:
    """Flag congestion when zone occupancy reaches its threshold."""
    def __init__(self, polygon: list[list[int]], threshold: int) -> None:
        if len(polygon) < 3 or threshold < 1:
            raise ValueError("congestion zone requires three points and a positive threshold")
        self.polygon, self.threshold = np.asarray(polygon, dtype=np.int32), threshold

    def check(self, vehicles: list[TrackedVehicle]) -> tuple[int, bool]:
        """Return current number inside the zone and congestion state."""
        occupancy = sum(cv2.pointPolygonTest(self.polygon, vehicle.centroid, False) >= 0 for vehicle in vehicles)
        return occupancy, occupancy >= self.threshold
