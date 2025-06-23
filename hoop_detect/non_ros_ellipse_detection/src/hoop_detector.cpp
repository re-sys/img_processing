#include "hoop_detector.hpp"

#include <numeric>

namespace non_ros_hoop {

namespace {
inline int makeOdd(int v) { return v % 2 == 0 ? v + 1 : v; }
}

cv::Mat HoopDetector::process(const cv::Mat &frame, std::vector<cv::Mat> *intermediates) {
  ellipse_ = cv::RotatedRect(); // reset

  if (frame.empty()) return frame.clone();

  // 1. HSV threshold ---------------------------------------------------------
  cv::Mat hsv, mask;
  cv::cvtColor(frame, hsv, cv::COLOR_BGR2HSV);
  cv::inRange(hsv,
              cv::Scalar(params_.h_lower, params_.s_lower, params_.v_lower),
              cv::Scalar(params_.h_upper, params_.s_upper, params_.v_upper),
              mask);

  // 2. Morphological opening -------------------------------------------------
  cv::Mat opened;
  int ksize = std::max(1, params_.morph_kernel);
  ksize = makeOdd(ksize);
  cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, {ksize, ksize});
  cv::morphologyEx(mask, opened, cv::MORPH_OPEN, kernel);

  // 3. Median blur -----------------------------------------------------------
  cv::Mat blurred;
  int bsize = makeOdd(std::max(1, params_.median_kernel));
  cv::medianBlur(opened, blurred, bsize);

  if (intermediates) {
    intermediates->clear();
    intermediates->push_back(mask.clone());
    intermediates->push_back(opened.clone());
    intermediates->push_back(blurred.clone());
  }

  // 4. Contour detection -----------------------------------------------------
  std::vector<std::vector<cv::Point>> contours;
  cv::findContours(blurred, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);

  double best_area = 0.0;
  cv::RotatedRect best_ellipse;

  for (const auto &c : contours) {
    double area = cv::contourArea(c);
    if (area < params_.min_area || area > params_.max_area) continue;
    if (c.size() < 5) continue; // fitEllipse requires >=5 points
    cv::RotatedRect e = cv::fitEllipse(c);
    if (area > best_area) {
      best_area = area;
      best_ellipse = e;
    }
  }

  cv::Mat debug = frame.clone();
  if (best_area > 0.0) {
    ellipse_ = best_ellipse;
    cv::ellipse(debug, ellipse_, {0, 255, 0}, 2);
  }

  return debug;
}

} // namespace non_ros_hoop 