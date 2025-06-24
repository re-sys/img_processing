#pragma once

#include <opencv2/opencv.hpp>
#include <vector>


namespace non_ros_hoop {

struct FilterParams {
  // Morphological opening kernel size (odd, >=1)
  int morph_kernel = 5;
  // Median blur kernel size (odd, >=1)
  int median_kernel = 9;

  // HSV threshold
  int h_lower = 2;
  int s_lower = 200;
  int v_lower = 54;

  int h_upper = 12;
  int s_upper = 255;
  int v_upper = 216;



  // RANSAC distance threshold (relative error on ellipse equation)     
  double ransac_thresh = 0.1; // |val-1| < thresh is inlier
  // RANSAC maximum iterations
  int ransac_iters = 1000;


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
   *   3: All contours (before过滤)  
   *   4: Filtered contours (after厚度/洞数/面积过滤)  
   */
  cv::Mat process(const cv::Mat &frame, std::vector<cv::Mat> *intermediates = nullptr);

  /* Access the last detected ellipse (may be empty => size == 0). */
  const cv::RotatedRect &lastEllipse() const { return ellipse_; }

private:
  FilterParams params_;
  cv::RotatedRect ellipse_;
};

} // namespace non_ros_hoop 