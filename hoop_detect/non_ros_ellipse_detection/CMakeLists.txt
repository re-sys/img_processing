cmake_minimum_required(VERSION 3.14)
project(non_ros_hoop_detector LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find OpenCV
find_package(OpenCV REQUIRED)

include_directories(
  ${OpenCV_INCLUDE_DIRS}
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# ----------------------------------------------------------------------------
# Hik camera wrapper
add_library(hik_camera
  src/hik_camera.cpp
)

target_link_libraries(hik_camera
  ${OpenCV_LIBS}
  MvCameraControl           # HIKVISION SDK (make sure this library is discoverable)
)

target_include_directories(hik_camera PUBLIC include)

# ----------------------------------------------------------------------------
# Hoop ellipse detector
add_library(hoop_detector
  src/hoop_detector.cpp
)

target_link_libraries(hoop_detector ${OpenCV_LIBS})

target_include_directories(hoop_detector PUBLIC include)

# ----------------------------------------------------------------------------
# Main interactive application
add_executable(hoop_detector_app src/main.cpp)

target_link_libraries(hoop_detector_app
  hik_camera
  hoop_detector
  ${OpenCV_LIBS}
)

# ----------------------------------------------------------------------------
# Installation
install(TARGETS hoop_detector_app RUNTIME DESTINATION bin) 