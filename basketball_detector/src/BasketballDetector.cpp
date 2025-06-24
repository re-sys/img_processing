#include "BasketballDetector.h"

using namespace cv;

BasketballDetector::BasketballDetector() = default;

void BasketballDetector::setMedianKernel(int k) {
    if (k % 2 == 0) ++k;            // enforce odd
    if (k < 3) k = 3;
    medianKernel = k;
}

void BasketballDetector::setMorphKernel(int k) {
    if (k < 1) k = 1;
    morphKernel = k;
}

static Mat _toBgr(const Mat &gray) {
    Mat bgr;
    if (gray.channels() == 1) {
        cvtColor(gray, bgr, COLOR_GRAY2BGR);
    } else {
        gray.copyTo(bgr);
    }
    return bgr;
}

// Draw label on top-left corner
Mat BasketballDetector::drawLabel(const Mat &img, const std::string &text) {
    Mat out = img.clone();
    putText(out, text, Point(10, 25), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0, 255, 0), 2);
    return out;
}

// Resize keeping aspect to desired size (fit inside)
Mat BasketballDetector::ensureSize(const Mat &src, const Size &size) {
    Mat dst;
    resize(src, dst, size);
    return dst;
}

DetectionResult BasketballDetector::process(const Mat &frame) {
    DetectionResult res;
    if (frame.empty()) return res;

    // Step 1: HSV mask
    Mat hsv;
    cvtColor(frame, hsv, COLOR_BGR2HSV);
    Mat mask;
    inRange(hsv, hsvLower, hsvUpper, mask);

    // Step 2: Median blur + closing
    setMedianKernel(medianKernel); // ensure odd
    Mat blurred;
    medianBlur(mask, blurred, medianKernel);

    setMorphKernel(morphKernel);
    Mat kernel = getStructuringElement(MORPH_RECT, Size(morphKernel, morphKernel));
    Mat closed;
    morphologyEx(blurred, closed, MORPH_CLOSE, kernel);

    // Step 3: Contours and circle fitting
    std::vector<std::vector<Point>> contours;
    findContours(closed, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    float maxScore = -1;
    Point2f bestCenter;
    float bestRadius = 0;

    for (const auto &cont : contours) {
        float area = contourArea(cont);
        if (area < 100) continue; // skip small noise
        Point2f center;
        float radius;
        minEnclosingCircle(cont, center, radius);

        // Score: prefer more circular ( area / (pi * r^2) ), closer to 1 and larger radius
        float circleArea = CV_PI * radius * radius;
        float circularity = area / circleArea;
        float score = circularity * radius; // heuristic
        if (score > maxScore) {
            maxScore = score;
            bestCenter = center;
            bestRadius = radius;
        }
    }

    Mat annotated = frame.clone();
    if (maxScore > 0) {
        circle(annotated, bestCenter, static_cast<int>(bestRadius), Scalar(0, 255, 0), 2);
        circle(annotated, bestCenter, 3, Scalar(0, 0, 255), -1);
        res.center = bestCenter;
        res.radius = bestRadius;
        res.found = true;
    }

    // Step 5: Compose debug view
    Size dispSize(320, 240);
    std::vector<Mat> row1, row2;
    row1.push_back(drawLabel(ensureSize(frame, dispSize), "Original"));
    row1.push_back(drawLabel(ensureSize(_toBgr(mask), dispSize), "Mask"));

    row2.push_back(drawLabel(ensureSize(_toBgr(closed), dispSize), "PostProc"));
    row2.push_back(drawLabel(ensureSize(annotated, dispSize), "Result"));

    Mat top, bottom;
    hconcat(row1, top);
    hconcat(row2, bottom);
    vconcat(top, bottom, res.composite);

    return res;
} 