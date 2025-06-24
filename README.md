# img_processing
sustech lab problem

## Overview
A collection of computer-vision utilities and experiments (OpenCV/C++ & Python) used in SUSTech lab projects. It spans hoop/chess detection modules, teaching labs, and handy scripts like the ArUco marker generator described below.

## Quick Start – Generate an ArUco marker
```bash
# Install dependency (OpenCV ≥4)
pip install opencv-python

# Create a 600×600 px marker (ID 7 from the 6×6 dictionary)
python generate_aruco_marker.py --dict DICT_6X6_250 --id 7 --size 600 --outfile marker_7 --jpg

# Set JPEG quality to 100 (maximum)
python generate_aruco_marker.py --dict DICT_4X4_50 --id 0 --size 6000 --svg-mm 600 \
                                --outfile aruco_60cm --jpg --jpg-quality 100

# Create a 60 cm × 60 cm SVG marker for large-scale printing
python generate_aruco_marker.py --dict DICT_6X6_250 --id 7 --outfile marker_7_large --svg-mm 600

# Outputs:
#   marker_7.png – high-resolution raster image (if --size specified)
#   marker_7_large.svg – 600 mm vector image ready for printing
```

## Build (Hoop Detector & HSV Calibrator)
```bash
cd hoop_detect/ellipse_detection
mkdir -p build && cd build
cmake ..
make -j
# 生成的可执行文件
./hoop_detector        # 实时篮筐检测
./hsv_calibrator       # 交互式 HSV 阈值标定
```
