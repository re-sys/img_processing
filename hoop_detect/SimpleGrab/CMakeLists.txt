cmake_minimum_required(VERSION 3.10)
project(SimpleGrab)

# 设置C++标准
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 查找OpenCV包
find_package(OpenCV REQUIRED)

# 添加海康相机SDK的头文件路径
include_directories(/opt/MVS/include)

# 添加可执行文件
add_executable(SimpleGrab SimpleGrab.cpp)

# 链接OpenCV库
target_link_libraries(SimpleGrab ${OpenCV_LIBS})

# 链接海康相机SDK库
target_link_libraries(SimpleGrab 
    /opt/MVS/lib/64/libMvCameraControl.so
) 