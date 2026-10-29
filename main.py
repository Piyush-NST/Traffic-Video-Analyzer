"""CLI entry point for Traffic Video Analyzer."""

from __future__ import annotations

import argparse
import sys

from analyzer.pipeline import TrafficAnalyzer


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze traffic video with YOLOv8 and ByteTrack.")
    parser.add_argument("--input", required=True, help="Source video path")
    parser.add_argument("--output", default="outputs/annotated.mp4", help="Annotated MP4 destination")
    parser.add_argument("--config", default="config.yaml", help="YAML settings path")
    args = parser.parse_args()
    try:
        result = TrafficAnalyzer(args.config).process(args.input, args.output)
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    print("Analysis complete")
    for label, path in (("Video", result.video_path), ("CSV", result.csv_path), ("Chart", result.chart_path), ("Summary", result.summary_path)):
        print(f"{label}: {path}")
    print(f"Frames: {result.processed_frames}/{result.total_frames}; counts: {result.counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
