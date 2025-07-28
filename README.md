# Video Processing Pipeline with Multiprocessing and OpenCV

This Python project demonstrates a multiprocessing pipeline for processing video files using OpenCV. The pipeline consists of three concurrent processes for streaming, decoding, and displaying video frames, enabling efficient real-time video processing.

## Installation
```bash
uv sync
```

## Run

run it with 
```bash
uv run main.py

| Argument | Description                                   | Usage              |
|----------|-----------------------------------------------|--------------------|
| `--path` | Path to the input video file.                 | `./<Vid_Name>.mp4` |
| `--blur` | Enable blurring of detected regions (flag).   | with / without     |

```