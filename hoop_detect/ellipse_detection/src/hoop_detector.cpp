#include "hoop_detector.hpp"

#include <numeric>
#include <cmath>
#include <random>
#include <unordered_set>

namespace non_ros_hoop {

namespace {
inline int makeOdd(int v) { return v % 2 == 0 ? v + 1 : v; }

// Convert grayscale images to BGR for consistent concatenation in debug mosaics.
inline cv::Mat toBGR(const cv::Mat &src) {
  if (src.channels() == 1) {
    cv::Mat dst;
    cv::cvtColor(src, dst, cv::COLOR_GRAY2BGR);
    return dst;
  }
  return src.clone();
}
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
    intermediates->push_back(toBGR(mask));
    intermediates->push_back(toBGR(opened));
    intermediates->push_back(toBGR(blurred));
  }

  // 4. Contour detection -----------------------------------------------------
  std::vector<std::vector<cv::Point>> contours_h;
  std::vector<cv::Vec4i> hierarchy;
  cv::findContours(blurred, contours_h, hierarchy, cv::RETR_CCOMP, cv::CHAIN_APPROX_NONE);

  // Visualisation image initialised black
  cv::Mat contoursVis = cv::Mat::zeros(frame.size(), CV_8UC3);

  // Compute area of each contour
  std::vector<double> areas(contours_h.size());
  for(size_t i=0;i<contours_h.size();++i){
    areas[i] = cv::contourArea(contours_h[i]);
  }

  // Determine dynamic threshold: 0.5 × second-largest contour area
  double area_thresh = 0.0;
  if(areas.size() >= 2){
    std::vector<double> sorted = areas;
    std::sort(sorted.begin(), sorted.end(), std::greater<>());
    area_thresh = 0.5 * sorted[1];
  }

  // Collect points only from large-enough contours
  std::vector<cv::Point2f> all_pts;
  for(size_t i=0;i<contours_h.size();++i){
    if(areas[i] < area_thresh) continue; // filter out small contours
    for(const auto &pt : contours_h[i]){
      all_pts.emplace_back(static_cast<float>(pt.x), static_cast<float>(pt.y));
    }
  }

  if(intermediates){
    intermediates->push_back(contoursVis);
  }

  cv::Mat debug = frame.clone();

  // Draw kept contours on contoursVis
  for(size_t i=0;i<contours_h.size();++i){
    if(areas[i] >= area_thresh){
      cv::drawContours(contoursVis, contours_h, static_cast<int>(i), {0,255,255}, 2);
    }
  }

  constexpr size_t MIN_INLIERS = 200;

  if (all_pts.size() >= 5) {
    // RANSAC ellipse fitting ------------------------------------------------
    size_t best_inliers = 0;
    cv::RotatedRect best_e;

    std::mt19937 rng(std::random_device{}());
    std::uniform_int_distribution<size_t> dist(0, all_pts.size() - 1);

    auto isInlier = [](const cv::Point2f &pt, const cv::RotatedRect &e, double tol) {
      // Convert point to ellipse-aligned coordinate frame
      double angle = e.angle * CV_PI / 180.0;
      double cosA = std::cos(angle);
      double sinA = std::sin(angle);

      double dx = pt.x - e.center.x;
      double dy = pt.y - e.center.y;

      double x_rot = dx * cosA + dy * sinA;
      double y_rot = -dx * sinA + dy * cosA;

      double a = e.size.width * 0.5;
      double b = e.size.height * 0.5;
      if (a < 1e-6 || b < 1e-6) return false;

      double val = (x_rot * x_rot) / (a * a) + (y_rot * y_rot) / (b * b);
      return std::abs(val - 1.0) < tol;
    };

    for (int iter = 0; iter < params_.ransac_iters; ++iter) {
      // Sample 5 distinct indices
      std::vector<cv::Point2f> sample;
      sample.reserve(5);
      std::unordered_set<size_t> idx_set;
      while (idx_set.size() < 5) {
        idx_set.insert(dist(rng));
      }
      for (auto idx : idx_set) sample.push_back(all_pts[idx]);

      // Fit ellipse on sample
      cv::RotatedRect e;
      try {
        e = cv::fitEllipse(sample);
      } catch (...) {
        continue; // degenerate sample
      }

      // Reject ellipses whose major axis exceeds half frame width
      double major_len = std::max(e.size.width, e.size.height);
      if (major_len > 0.5 * frame.cols) {
        continue; // unrealistic candidate
      }

      // Count inliers
      size_t inliers = 0;
      for (const auto &p : all_pts) {
        if (isInlier(p, e, params_.ransac_thresh)) ++inliers;
      }

      if (inliers > best_inliers) {
        best_inliers = inliers;
        best_e = e;
      }
    }

    // Refit using all inliers of best model for better accuracy
    if (best_inliers > 0) {
      std::vector<cv::Point2f> inlier_pts;
      inlier_pts.reserve(best_inliers);
      for (const auto &p : all_pts) {
        if (isInlier(p, best_e, params_.ransac_thresh)) inlier_pts.push_back(p);
      }
      if (inlier_pts.size() >= 5) {
        best_e = cv::fitEllipse(inlier_pts);
      }

      ellipse_ = best_e;
      // Validate ellipse major axis not exceeding half frame width ----------
      double major = std::max(ellipse_.size.width, ellipse_.size.height);
      if (major > 0.5 * frame.cols) {
        ellipse_ = cv::RotatedRect(); // invalidate
      }

      if (ellipse_.size.width > 0 && ellipse_.size.height > 0 && inlier_pts.size() >= MIN_INLIERS) {
        // Draw ellipse
        cv::ellipse(debug, ellipse_, {0, 255, 0}, 2);

        // Highlight inlier points in magenta
        for(const auto &p : inlier_pts){
          cv::circle(debug, p, 2, {255,0,255}, -1);
        }

        // Overlay inlier count (green if sufficient, red otherwise)
        cv::Scalar col = inlier_pts.size() >= MIN_INLIERS ? cv::Scalar(0,255,0) : cv::Scalar(0,0,255);
        std::string text = "inliers: " + std::to_string(inlier_pts.size());
        cv::putText(debug, text, {10, 30}, cv::FONT_HERSHEY_SIMPLEX, 1.0, col, 2);
      }
    }
  }

  // Append final debug frame for consumers that need full pipeline view
  if(intermediates){
    intermediates->push_back(debug.clone());
  }

  return debug;
}

} // namespace non_ros_hoop 