#pragma once

#include <opencv2/opencv.hpp>
#include <vector>

namespace non_ros_hoop {

struct FilterParams {
  // Morphological opening kernel size (odd, >=1)
  int morph_kernel = 5;
  // Median blur kernel size (odd, >=1)
  int median_kernel = 5;

  // HSV threshold
  int h_lower = 0;
  int s_lower = 0;
  int v_lower = 0;
  int h_upper = 179;
  int s_upper = 255;
  int v_upper = 255;

  // Contour area filtering
  double min_area = 500.0;
  double max_area = 1e6;

  // RANSAC distance threshold (pixels)
  double ransac_thresh = 5.0;
};

class HoopDetector {
public:
  HoopDetector() = default;

  void setParams(const FilterParams &p) { params_ = p; }
  const FilterParams &params() const { return params_; }

  /* Process one frame and return a debug image with the detected ellipse drawn.  
   * If intermediates is non-null, it will be filled with:  
   *   0: HSV threshold mask  
   *   1: Morphologically opened mask  
   *   2: Median-filtered mask  
   */
  cv::Mat process(const cv::Mat &frame, std::vector<cv::Mat> *intermediates = nullptr);

  /* Access the last detected ellipse (may be empty => size == 0). */
  const cv::RotatedRect &lastEllipse() const { return ellipse_; }

private:
  FilterParams params_;
  cv::RotatedRect ellipse_;
};

} // namespace non_ros_hoop 