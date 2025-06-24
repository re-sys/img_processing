cmake_minimum_required(VERSION 3.14)
project(hoop_detector LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# -----------------------------------------------------------------------------
# Dependencies
# -----------------------------------------------------------------------------
find_package(OpenCV REQUIRED)

# -----------------------------------------------------------------------------
# Include directories – OpenCV + HikVision SDK + project headers
# -----------------------------------------------------------------------------
include_directories(
  /opt/MVS/include                        # HikVision MVS SDK headers
  ${CMAKE_CURRENT_SOURCE_DIR}/include     # Project headers (hik_camera.hpp ...)
  ${OpenCV_INCLUDE_DIRS}
)

# -----------------------------------------------------------------------------
# Source files grouped by functionality
# -----------------------------------------------------------------------------
set(HIK_SRC src/hik_camera.cpp)
set(HOOP_SRC src/hoop_detector.cpp)

# -----------------------------------------------------------------------------
# Libraries (static)
# -----------------------------------------------------------------------------
add_library(hik_camera STATIC ${HIK_SRC})
add_library(hoop_logic  STATIC ${HOOP_SRC})

# Link libraries internally
target_link_libraries(hik_camera PUBLIC ${OpenCV_LIBS} /opt/MVS/lib/64/libMvCameraControl.so)

# hoop_logic only needs OpenCV
target_link_libraries(hoop_logic PUBLIC ${OpenCV_LIBS})

target_include_directories(hik_camera PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include /opt/MVS/include)
target_include_directories(hoop_logic  PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

# -----------------------------------------------------------------------------
# Executable: hoop_detector (main application)
# -----------------------------------------------------------------------------
add_executable(hoop_detector src/main.cpp)

target_link_libraries(hoop_detector
  hoop_logic
  hik_camera
  ${OpenCV_LIBS}
  /opt/MVS/lib/64/libMvCameraControl.so
)

# -----------------------------------------------------------------------------
# Executable: hsv_calibrator (interactive HSV tuning)
# -----------------------------------------------------------------------------
add_executable(hsv_calibrator src/hsv_calibrator.cpp)

target_link_libraries(hsv_calibrator
  hik_camera
  ${OpenCV_LIBS}
  /opt/MVS/lib/64/libMvCameraControl.so
)

# ----------------------------------------------------------------------------
# Installation
install(TARGETS hoop_detector hsv_calibrator RUNTIME DESTINATION bin) 