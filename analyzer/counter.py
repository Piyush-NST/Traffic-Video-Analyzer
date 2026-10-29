"""Directional line-crossing counting."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from math import hypot

from .tracker import TrackedVehicle


@dataclass(frozen=True)
class CrossingEvent:
    """A one-time line crossing."""
    track_id: int
    class_name: str
    direction: str


class LineCounter:
    """Count each tracked vehicle once as it crosses a directed line."""
    def __init__(self, start: list[int], end: list[int], in_direction: str, min_distance_px: float) -> None:
        if len(start) != 2 or len(end) != 2 or start == end:
            raise ValueError("counting line points must be distinct [x, y] values")
        if in_direction not in {"positive_to_negative", "negative_to_positive"} or min_distance_px < 0:
            raise ValueError("invalid counting configuration")
        self.start, self.end, self.in_direction, self.min_distance_px = tuple(start), tuple(end), in_direction, min_distance_px
        self.previous: dict[int, tuple[float, float]] = {}
        self.counted_ids: set[int] = set()
        self.counts: dict[str, Counter[str]] = defaultdict(Counter)

    def _side(self, point: tuple[float, float]) -> float:
        """Return signed perpendicular distance from the directed infinite line."""
        ax, ay = self.start
        bx, by = self.end
        px, py = point
        return ((bx - ax) * (py - ay) - (by - ay) * (px - ax)) / hypot(bx - ax, by - ay)

    def _crosses_segment(self, previous: tuple[float, float], current: tuple[float, float], before: float, after: float) -> bool:
        """Ensure the crossing falls on the finite configured line segment."""
        ratio = abs(before) / (abs(before) + abs(after))
        crossing_x = previous[0] + ratio * (current[0] - previous[0])
        crossing_y = previous[1] + ratio * (current[1] - previous[1])
        ax, ay = self.start
        bx, by = self.end
        tolerance = self.min_distance_px
        return min(ax, bx) - tolerance <= crossing_x <= max(ax, bx) + tolerance and min(ay, by) - tolerance <= crossing_y <= max(ay, by) + tolerance

    def update(self, vehicles: list[TrackedVehicle]) -> list[CrossingEvent]:
        """Return newly observed crossings."""
        events: list[CrossingEvent] = []
        for vehicle in vehicles:
            current = vehicle.centroid
            previous = self.previous.get(vehicle.track_id)
            self.previous[vehicle.track_id] = current
            if previous is None or vehicle.track_id in self.counted_ids:
                continue
            before, after = self._side(previous), self._side(current)
            if abs(before) < self.min_distance_px or abs(after) < self.min_distance_px or before * after >= 0 or not self._crosses_segment(previous, current, before, after):
                continue
            positive_to_negative = before > 0 and after < 0
            is_in = positive_to_negative == (self.in_direction == "positive_to_negative")
            direction = "in" if is_in else "out"
            self.counted_ids.add(vehicle.track_id)
            self.counts[direction][vehicle.class_name] += 1
            events.append(CrossingEvent(vehicle.track_id, vehicle.class_name, direction))
        return events

    def totals(self) -> dict[str, dict[str, int]]:
        """Return serializable counts by direction and class."""
        return {direction: dict(values) for direction, values in self.counts.items()}
