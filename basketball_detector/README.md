# Basketball Detector

基于 OpenCV 的简单篮球检测示例，使用 **HSV 阈值**、**中值滤波**、**形态学闭运算** 与 **最小外接圆拟合**，在视频流中实时检测并标记篮球位置。

## 功能简介
1. HSV 阈值 (H: 0-19, S: 154-255, V: 20-83) 提取篮球颜色掩码。
2. 中值滤波抑制噪声，可在暂停模式下通过 TrackBar 动态调整核大小。
3. 形态学闭运算填补空洞，同样支持动态调整核大小。
4. 轮廓提取后，选择最像篮球的目标并拟合最小外接圆，绘制圆边界与圆心。
5. 同时展示四张调试视图（原图、掩码、后处理、最终结果）拼接在一个窗口中，方便比对。
6. 支持 **q** 键暂停/继续；暂停状态下拖动 TrackBar 会立即刷新检测结果，实现所见即所得；按 **e** 或 **ESC** 退出。

## 环境依赖
- C++17
- OpenCV ≥ 4.x （含 `opencv_videoio`, `opencv_highgui`, `opencv_imgproc` 等模块）
- CMake ≥ 3.10

## 构建与运行
```bash
# 克隆或进入本项目
cd basketball_detector

# 创建构建目录并编译
mkdir build && cd build
cmake ..
make -j4   # 根据 CPU 核心数调整

# 运行（将 <video.avi> 替换为你的视频路径）
./basketball_detector <video.avi>
```

> 若系统中存在多个 OpenCV 版本，可通过 `cmake .. -DOpenCV_DIR=/your/opencv/lib/cmake/opencv4` 指定具体版本。

## 目录结构
```
basketball_detector/
├── include/              # 头文件
│   └── BasketballDetector.h
├── src/                  # 源文件
│   ├── BasketballDetector.cpp
│   └── main.cpp
├── CMakeLists.txt        # 构建脚本
└── README.md             # 当前文档
```

## 参考
- OpenCV 官方文档 <https://docs.opencv.org/> 