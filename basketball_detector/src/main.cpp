#include "BasketballDetector.h"
#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;

int main(int argc, char **argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <video.avi>" << std::endl;
        return -1;
    }

    std::string videoPath = argv[1];
    VideoCapture cap(videoPath);
    if (!cap.isOpened()) {
        std::cerr << "Failed to open video: " << videoPath << std::endl;
        return -1;
    }

    BasketballDetector detector;

    const std::string winName = "Basketball Detection";
    namedWindow(winName, WINDOW_NORMAL);

    bool paused = false;

    // Control window for parameter tuning
    const std::string ctrlWin = "Controls";
    namedWindow(ctrlWin, WINDOW_NORMAL);
    int medianK = 5;
    int morphK = 5;
    createTrackbar("Median Kernel", ctrlWin, &medianK, 25);
    createTrackbar("Morph Kernel", ctrlWin, &morphK, 25);

    Mat frame;
    while (true) {
        if (!paused) {
            cap >> frame;
            if (frame.empty()) break;
        }

        // Always update parameters from trackbar (for live refresh when paused)
        medianK = getTrackbarPos("Median Kernel", ctrlWin);
        morphK = getTrackbarPos("Morph Kernel", ctrlWin);

        detector.setMedianKernel(medianK);
        detector.setMorphKernel(morphK);

        DetectionResult res = detector.process(frame);
        imshow(winName, res.composite.empty() ? frame : res.composite);

        // Use short delay so GUI remains responsive and parameters refresh even when paused
        char c = (char)waitKey(10);
        if (c == 'e' || c == 27) break; // quit on e or ESC
        if (c == 'q') paused = !paused; // toggle pause on q
    }

    destroyAllWindows();
    return 0;
} 