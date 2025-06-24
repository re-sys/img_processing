#ifndef BASKETBALL_DETECTOR_H
#define BASKETBALL_DETECTOR_H

#include <opencv2/opencv.hpp>
#include <vector>

struct DetectionResult {
    cv::Mat composite;      // Concatenated view of intermediate steps
    cv::Point2f center;     // Detected ball center (if any)
    float radius{0.f};      // Detected ball radius
    bool found{false};      // Was a ball detected?
};

class BasketballDetector {
public:
    BasketballDetector();

    // Process a single frame and return result (with composite preview)
    DetectionResult process(const cv::Mat &frame);

    // Parameter setters (values are clamped/sanitised internally)
    void setMedianKernel(int k);
    void setMorphKernel(int k);

private:
    // HSV range for orange basketball (default constants)
    const cv::Scalar hsvLower{0, 154, 20};   // (H, S, V)
    const cv::Scalar hsvUpper{19, 255, 83};

    int medianKernel{5}; // must be odd >=3
    int morphKernel{5};  // size of square kernel for closing

    // Helper functions
    static cv::Mat drawLabel(const cv::Mat &img, const std::string &text);
    static cv::Mat ensureSize(const cv::Mat &src, const cv::Size &size);
};

#endif // BASKETBALL_DETECTOR_H 