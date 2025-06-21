#pragma once

#include <opencv2/opencv.hpp>
#include "MvCameraControl.h"

namespace basket_hoop_detector
{
class HikCamera
{
public:
  HikCamera();
  ~HikCamera();

  bool open();
  bool getFrame(cv::Mat &frame);
  void close();

private:
  void *handle_ = nullptr;  // camera handle
};
}  // namespace basket_hoop_detector 