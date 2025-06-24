cmake_minimum_required(VERSION 3.10)
project(basketball_detector)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV REQUIRED)

include_directories(include ${OpenCV_INCLUDE_DIRS})

file(GLOB SRC_FILES src/*.cpp)
add_executable(basketball_detector ${SRC_FILES})

target_link_libraries(basketball_detector ${OpenCV_LIBS}) 