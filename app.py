"""Streamlit user interface for Traffic Video Analyzer."""

from __future__ import annotations

import copy
import tempfile
from pathlib import Path

import streamlit as st

from analyzer.pipeline import TrafficAnalyzer, load_config

st.set_page_config(page_title="Traffic Video Analyzer", page_icon="🚦", layout="wide")
st.title("🚦 Traffic Video Analyzer")
st.caption("YOLOv8 detection, ByteTrack IDs, directional counts, speed estimates, and congestion alerts.")

try:
    base_config = load_config("config.yaml")
except (FileNotFoundError, ValueError) as error:
    st.error(f"Unable to load configuration: {error}")
    st.stop()

with st.sidebar:
    st.header("Analysis settings")
    ui = base_config["ui"]
    confidence = st.slider("Detection confidence", float(ui["confidence_min"]), float(ui["confidence_max"]), float(base_config["model"]["confidence_threshold"]), float(ui["confidence_step"]))
    start, end = base_config["counting"]["line_start"], base_config["counting"]["line_end"]
    st.caption("Counting-line coordinates (pixels)")
    a, b = st.columns(2)
    line_x1, line_y1 = a.number_input("Line start X", min_value=0, value=int(start[0])), b.number_input("Line start Y", min_value=0, value=int(start[1]))
    c, d = st.columns(2)
    line_x2, line_y2 = c.number_input("Line end X", min_value=0, value=int(end[0])), d.number_input("Line end Y", min_value=0, value=int(end[1]))

uploaded = st.file_uploader("Upload a traffic video", type=["mp4", "avi", "mov", "mkv"])
if not uploaded:
    st.info("Upload a traffic video to begin.")
elif st.button("Analyze video", type="primary"):
    config = copy.deepcopy(base_config)
    config["model"]["confidence_threshold"] = confidence
    config["counting"]["line_start"] = [int(line_x1), int(line_y1)]
    config["counting"]["line_end"] = [int(line_x2), int(line_y2)]
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded.name).suffix) as file:
        file.write(uploaded.getbuffer())
        source = Path(file.name)
    output = Path("outputs") / f"{Path(uploaded.name).stem}_annotated.mp4"
    bar, status = st.progress(0, text="Starting analysis…"), st.empty()
    def progress(done: int, total: int) -> None:
        bar.progress(min(done / max(total, 1), 1.0), text=f"Analyzing frame {done:,} of {total:,}")
        status.caption(f"Processed {done:,} frames")
    try:
        result = TrafficAnalyzer(config=config).process(source, output, progress)
        bar.progress(1.0, text="Complete")
        st.success(f"Processed {result.processed_frames:,} frames. Counts: {result.counts}")
        st.video(str(result.video_path))
        one, two, three = st.columns(3)
        one.download_button("Download annotated video", result.video_path.read_bytes(), result.video_path.name, "video/mp4")
        two.download_button("Download crossing CSV", result.csv_path.read_bytes(), result.csv_path.name, "text/csv")
        three.download_button("Download summary chart", result.chart_path.read_bytes(), result.chart_path.name, "image/png")
        st.image(str(result.chart_path), caption="Traffic summary")
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        st.error(f"Analysis failed: {error}")
    finally:
        source.unlink(missing_ok=True)
