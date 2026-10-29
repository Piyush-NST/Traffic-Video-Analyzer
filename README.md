# Traffic Video Analyzer

This project detects cars, buses, trucks, and motorcycles with YOLOv8; assigns ByteTrack identities; counts directed virtual-line crossings; estimates calibrated speed; and warns when a polygon zone is congested. Every run creates an annotated MP4, crossing CSV, summary PNG, and JSON summary.

## Setup and run

```bash
# Python 3.10–3.13 is recommended for the PyTorch/YOLO stack.
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python main.py --input traffic.mp4 --output outputs/annotated.mp4
streamlit run app.py
```

The first run automatically downloads `yolov8n.pt`. `ffmpeg` enables H.264 browser re-encoding; without it, the OpenCV MP4 is retained. All tunable operational values live in `config.yaml`: model/tracker thresholds, line endpoints and direction, calibration, polygon/threshold, encoder, and optional test frame limit.

## How it works

YOLOv8 detects the configured vehicle classes on every frame. ByteTrack associates detections across frames. A centroid crossing the directed line emits one `in` or `out` event per ID. Centroid displacement, FPS, and `pixels_per_meter` produce a smoothed speed. Vehicles whose centroids fall inside the configured polygon determine congestion. The pipeline draws all live overlays and exports the report artifacts.

The CSV columns are `timestamp,vehicle_id,class,speed_kmh,direction`; each row is a line-crossing event. The PNG has vehicles per minute and class distribution.

## Calibration guide

1. Find a flat roadway segment at roughly the same depth as the vehicles.
2. Measure a known real distance, such as a 10 m lane-marking segment.
3. Measure the same segment in pixels on a video frame.
4. Set `pixels_per_meter = pixels / metres` in `config.yaml`.
5. Compare several results against known speeds and tune the value. Use a perspective homography for serious measurement.

## Screenshots / generated artifacts

After processing, view `outputs/annotated.mp4` for overlays, `outputs/annotated.csv` for events, and `outputs/annotated_summary.png` for the report chart.

## Known limitations

- Small, occluded, nighttime, or unusual vehicles may be missed or misclassified.
- Heavy occlusion can cause ID switches and affect counts/speeds.
- A single pixel-to-metre value is approximate under perspective and camera motion.
- Each ID is counted only once by the one virtual line.
- Long 4K sources are much faster with a suitable GPU.

## Future improvements

1. License plate recognition with privacy controls.
2. Wrong-way driving and illegal-turn detection.
3. Homography-based real-world speed calibration.
4. Multiple lane lines and turning-movement matrices.
5. Live camera ingestion with database/dashboard reporting.
