#pragma once

#include <opencv2/opencv.hpp>
#include "MvCameraControl.h"

namespace non_ros_hoop {

class HikCamera {
public:
  HikCamera();
  ~HikCamera();

  /* Open the first available HikVision camera.  
   * Returns true on success. */
  bool open();

  /* Retrieve one frame from the camera as a BGR cv::Mat.  
   * Returns true if a frame was successfully captured. */
  bool getFrame(cv::Mat &frame);

  /* Close the camera and release resources. */
  void close();

private:
  void *handle_ = nullptr;  // Camera SDK handle
};

}  // namespace non_ros_hoop 