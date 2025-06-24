# basket_hoop_detector

A ROS 2 package that performs real-time basketball-hoop localisation with a Hikvision camera. The workflow is split into **three** standalone utilities so that colour segmentation, camera calibration and detection can run independently.

---

## 1. Package build & installation

```bash
# ➊ Clone (or copy) into your ROS 2 workspace
cd ~/ros2_ws/src
# git clone <your-repo-url>

# ➋ Build only this package (Humble / Iron / Rolling)
cd ..
colcon build --packages-select basket_hoop_detector --symlink-install

# ➌ Source the overlay
source install/setup.bash
```

> **NOTE 1**   The package links to the Hikvision *MvCameraControl* SDK. If the SDK is **not** installed to `/opt/MVS`, open `CMakeLists.txt` and change
> ```cmake
> set(HIKVISION_SDK_INCLUDE_DIR "/custom/path/include")
> set(HIKVISION_SDK_LIB_DIR     "/custom/path/lib")
> ```
>
> **NOTE 2**   Make sure the user has permission to access the camera device (e.g. `sudo chmod a+rw /dev/…`).

---

## 2. Directory layout
```
basket_hoop_detector
├── include/…            # HikCamera wrapper
├── src/
│   ├── hik_camera.cpp   # camera class implementation
│   ├── hsv_calibrator.cpp   # colour threshold wizard
│   ├── camera_calibrator.cpp# intrinsic calibration & image saver
│   └── hoop_detector_node.cpp # main detection node
├── config/hsv_config.yaml   # template configuration
├── launch/hoop_detector.launch.py
└── README.md                # this file
```

---

## 3. Applications (recommended顺序)

### 3.1 hsv_calibrator
Interactive tool to sample HSV pixels and produce a configuration YAML.

```bash
ros2 run basket_hoop_detector hsv_calibrator   # default writes config/hsv_config.yaml
# OR specify output path
ros2 run basket_hoop_detector hsv_calibrator /tmp/my_hoop.yaml
```
操作：
1. 左键多次点击篮筐像素 → 自动更新上下阈值；
2. `s` 保存 YAML 后退出；`q` 直接退出不保存。

生成的 YAML 片段示例：
```yaml
hsv:
  lower: [20, 80, 80]
  upper: [40, 255, 255]
ransac:
  iterations: 200
  tolerance: 0.05
```

### 3.2 camera_calibrator
Utility to capture chessboard pictures **(s key)**, calibrate intrinsic parameters **(c key)** and append results to the same YAML so that the detector can use accurate camera matrix / distortion.

```bash
ros2 run basket_hoop_detector camera_calibrator        # pictures/ & config/hsv_config.yaml default paths
# 自定义保存目录 & YAML
ros2 run basket_hoop_detector camera_calibrator ~/pics ~/my_hoop.yaml
```
要求：最少 5 张 9×6 棋盘格图片（方格边长默认 2.5 cm，可在源码中修改）。

当标定完成后，YAML 会附加：
```yaml
camera:
  matrix: [fx, 0, cx, 0, fy, cy, 0, 0, 1]
  dist:   [k1, k2, p1, p2, k3]
```

### 3.3 hoop_detector_node (main)
Real-time detection + pose estimation node.

启动（使用 launch）：
```bash
ros2 launch basket_hoop_detector hoop_detector.launch.py   # 默认读取 config/hsv_config.yaml
```

常用参数（可通过 `--ros-args -p` 覆盖）：
* `debug_level` 0~2 – 视觉调试窗口与日志粒度
* `config_file` – 指定 YAML 路径
* `hsv_lower / hsv_upper` – 直接用 ROS 参数覆写阈值
* `ransac_iterations_param / ransac_tolerance_param` – 覆写 RANSAC 参数

输出：
1. TF `camera_link -> basket_hoop` — 在 RViz2 中可视化；
2. `/debug_image` (sensor_msgs/Image) — 调试图话题。

---

## 4. Typical workflow
1. **颜色标定** `hsv_calibrator` → 保存 `hsv_config.yaml`。
2. **相机内参** 可选：`camera_calibrator` → 同一 YAML 写入 `camera:` 部分。
3. **检测** `hoop_detector_node` （通过 launch 或直接节点调用）。
4. **RViz2 可视化** `rviz2` → Fixed Frame 设为 `camera_link` → 添加 TF、Image 等显示。

---

## 5. Troubleshooting
| 问题 | 解决方法 |
|------|----------|
| 无法连接相机 | 检查 MVS SDK & 设备权限；确认 `CMakeLists.txt` 路径 |
| 编译缺少头文件 | `sudo apt install ros-$ROS_DISTRO-vision-opencv libyaml-cpp-dev` |
| 无法找到棋盘格 | 提高环境光、确保棋盘完整入镜；或调整 `board_size/square_size` |
| TF 不显示 | 确认 `/tf` 有数据，Fixed Frame= `camera_link` |

---

Made with ❤️  by **basket_hoop_detector** contributors. 