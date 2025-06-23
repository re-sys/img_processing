# non_ros_hoop_detector

This folder contains a standalone (non-ROS) demo program for detecting a basketball hoop (approximated as an ellipse) from a HikVision industrial camera in real-time.

## Directory layout

```
include/
  hik_camera.hpp      – Thin C++ wrapper around the HikVision MVS SDK
  hoop_detector.hpp   – Image-processing & ellipse-fitting logic
src/
  hik_camera.cpp      – HikVision camera wrapper implementation
  hoop_detector.cpp   – Ellipse detector implementation (OpenCV)
  main.cpp            – Interactive application (keyboard & GUI)
CMakeLists.txt        – Build configuration
```

## Features

* Real-time acquisition from the first detected Hik camera.
* HSV threshold + morphology + median filter + contour filtering.
* Ellipse fitting on the largest valid contour.
* Debug windows:
  * `r` – Start recording the raw camera stream to **MJPG / AVI** (`q` to stop).
  * `c` – Freeze the current frame and open a **control panel** with track-bars to tune:
    * Morphological opening kernel size.
    * Median blur kernel size.
    * HSV lower / upper thresholds (H, S, V).
    * Contour area limits.
    * RANSAC/ellipse distance threshold.
    * All intermediate images (`mask`, `opened`, `blurred`) are displayed and updated in real-time.
  * `q` – Quit the single-frame debug mode and resume live processing.
* `ESC` to quit the application.

## Building

```bash
mkdir build && cd build
cmake ..
make -j
./hoop_detector_app
```

> **Note**  The HIKVision **MVS** SDK must be installed and its `MvCameraControl` library discoverable by the dynamic loader. Adjust the `target_link_libraries` line in `CMakeLists.txt` if your library name/path differs. 