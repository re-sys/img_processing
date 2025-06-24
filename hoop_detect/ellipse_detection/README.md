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
* HSV threshold → morphology → median blur → contour quality filtering → **RANSAC 椭圆拟合**。
* 轮廓滤波:
  * 面积范围 (`min_area`, `max_area`).
  * **洞数** (`max_holes`) – 内部洞(子轮廓)多于该值则忽略。
  * **厚度** (`min_thickness`) – 近似 `area / perimeter`，过细的轮廓被丢弃。
  * 椭圆长轴必须小于图像宽度的一半才能被接受。
* GUI/键盘交互：
  * `p` – 暂停 / 继续。暂停时会冻结当前帧并弹出 **control** 窗口，可实时调以下参数：
    * `morph_kernel` – 开运算核大小。
    * `median_kernel` – 中值滤波核大小。
    * `max_holes` – 允许的最大洞数。
    * `thickness` – 轮廓最小厚度阈值(像素)。
  * `v` – 开始 / 结束录像 (MJPG-AVI)。
  * `q` / `ESC` – 退出程序。
* 调参时，将同步显示以下 Mosaic：
  * `debug` ‑ 原图叠加检测椭圆。
  * `mask` ‑ HSV 阈值分割图。
  * `opened` ‑ 形态学开运算结果。
  * `blurred` ‑ 中值滤波结果。
  * `all_cnt` ‑ 输入所有外轮廓可视化（黄色）。
  * `filt_cnt` ‑ 通过面积/洞数/厚度过滤后的轮廓（红色）。

## Building

```bash
mkdir build && cd build
cmake ..
make -j
./hoop_detector_app
```

> **Note**  The HIKVision **MVS** SDK must be installed and its `MvCameraControl` library discoverable by the dynamic loader. Adjust the `target_link_libraries` line in `CMakeLists.txt` if your library name/path differs. 

## Basketball detector demo

### Keys & interaction

* `p` – Pause / resume video playback. When paused:
  * The current frame is frozen and a **control window** pops up with four track-bars to tune `h_lower`, `h_upper`, `s_lower`, `s_upper`.
  * Click anywhere on the frozen image to print the pixel's HSV values in the terminal – handy for threshold selection.
* `ESC` – Quit application.

### Build & run

```bash
mkdir -p build && cd build
cmake ..
make -j           # builds both hoop_detector_app and ball_detector_app

# 1) Live camera (default 0)
./ball_detector_app            # or ./ball_detector_app 0

# 2) Specific video file / stream
./ball_detector_app basketball_clip.mp4

# 3) Single image
./ball_detector_app -i test.jpg
```

### Typical workflow

1. Launch with a test video or webcam showing a basketball.
2. Press `p` to freeze the frame.
3. Move the `H / S` sliders until the mask highlights only the basketball (watch the intermediate views).
4. Confirm that a green circle + red cross marks the centre.
5. Press `p` again to resume live detection with the tuned parameters.

## HSV Calibrator (`hsv_calibrator`)

用于快速取得 HSV 阈值的辅助小工具。

```bash
./hsv_calibrator          # 连接第一台 Hik 摄像机
```

操作说明：

* `p` 暂停 / 继续；暂停后可点击像素。
* 左键单击 采样该像素的 HSV，程序会自动累积 min/max 并在终端实时打印。
* `c` 清空现有采样。
* `q` / `ESC` 退出并在终端输出最终的 `lower` / `upper` 数值，直接可复制到 YAML/代码中。

窗口左侧显示原图，右侧显示根据当前阈值生成的二值 mask，便于验证分割效果。

--- 