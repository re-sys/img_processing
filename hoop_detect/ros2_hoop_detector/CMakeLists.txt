cmake_minimum_required(VERSION 3.8)
project(basket_hoop_detector)

# Default to C++17
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(image_transport REQUIRED)
find_package(OpenCV REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(yaml-cpp REQUIRED)

# Hikvision SDK path - adjust these if needed
set(HIKVISION_SDK_INCLUDE_DIR "/opt/MVS/include")
set(HIKVISION_SDK_LIB_DIR "/opt/MVS/lib/64")

include_directories(
  include
  ${HIKVISION_SDK_INCLUDE_DIR}
  ${OpenCV_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
)

link_directories(
  ${HIKVISION_SDK_LIB_DIR}
)

add_library(hik_camera
  src/hik_camera.cpp
)

target_link_libraries(hik_camera
  ${OpenCV_LIBS}
  ${HIKVISION_SDK_LIB_DIR}/libMvCameraControl.so
)

add_executable(hoop_detector_node src/hoop_detector_node.cpp)

ament_target_dependencies(hoop_detector_node
  rclcpp sensor_msgs std_msgs cv_bridge image_transport tf2 tf2_ros geometry_msgs
)

target_link_libraries(hoop_detector_node
  hik_camera
  ${OpenCV_LIBS}
  ${HIKVISION_SDK_LIB_DIR}/libMvCameraControl.so
  yaml-cpp
)

install(TARGETS hoop_detector_node hik_camera
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY include/
  DESTINATION include
)

install(DIRECTORY launch/
  DESTINATION share/${PROJECT_NAME}/launch
)

install(DIRECTORY config/
  DESTINATION share/${PROJECT_NAME}/config
)

# HSV calibrator (stand-alone, non-ROS)
add_executable(hsv_calibrator src/hsv_calibrator.cpp)
target_link_libraries(hsv_calibrator
  hik_camera
  ${OpenCV_LIBS}
  ${HIKVISION_SDK_LIB_DIR}/libMvCameraControl.so
  yaml-cpp
)
install(TARGETS hsv_calibrator
  DESTINATION lib/${PROJECT_NAME}
)

# Camera intrinsic calibrator
add_executable(camera_calibrator src/camera_calibrator.cpp)
target_link_libraries(camera_calibrator
  hik_camera
  ${OpenCV_LIBS}
  ${HIKVISION_SDK_LIB_DIR}/libMvCameraControl.so
  yaml-cpp
)
install(TARGETS camera_calibrator
  DESTINATION lib/${PROJECT_NAME}
)

ament_package() 