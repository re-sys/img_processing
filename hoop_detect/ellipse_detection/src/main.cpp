#include "hik_camera.hpp"
#include "hoop_detector.hpp"

#include <opencv2/opencv.hpp>
#include <chrono>
#include <filesystem>
#include <iomanip>
#include <sstream>
#include <vector>
#include <string>
#include <algorithm>

using namespace non_ros_hoop;
namespace fs = std::filesystem;

static std::string timestampString() {
  auto now = std::chrono::system_clock::now();
  auto tt = std::chrono::system_clock::to_time_t(now);
  std::tm tm;
#ifdef _WIN32
  localtime_s(&tm, &tt);
#else
  localtime_r(&tt, &tm);
#endif
  std::stringstream ss;
  ss << std::put_time(&tm, "%Y%m%d_%H%M%S");
  return ss.str();
}

// Global detector params used by trackbar callbacks
static FilterParams g_params;
static HoopDetector g_detector;
static cv::Mat g_frozen;

static const std::string WINDOW_NAME = "display";
static const int DISP_W = 1280;
static const int DISP_H = 720;

static cv::Mat fitToWindow(const cv::Mat &src){
  if(src.empty()) return src;
  double scale = std::min(1.0, std::min(static_cast<double>(DISP_W)/src.cols,
                                        static_cast<double>(DISP_H)/src.rows));
  if(scale >= 1.0) return src; // already fits
  cv::Mat dst;
  cv::resize(src, dst, cv::Size(), scale, scale);
  return dst;
}

static cv::Mat addLabel(const cv::Mat &img, const std::string &label){
  cv::Mat out;
  if(img.channels()==1)
    cv::cvtColor(img, out, cv::COLOR_GRAY2BGR);
  else
    out = img.clone();
  cv::putText(out, label, {10,30}, cv::FONT_HERSHEY_SIMPLEX, 1.0, {0,0,255}, 2);
  return out;
}

static cv::Mat buildMosaic(const std::vector<std::pair<cv::Mat,std::string>> &imgs, int cols = 2) {
  if (imgs.empty()) return cv::Mat();
  std::vector<cv::Mat> processed;
  const int target_h = 300; // resize height for each tile
  for (const auto &p : imgs) {
    cv::Mat labeled = addLabel(p.first, p.second);
    double scale = static_cast<double>(target_h) / labeled.rows;
    cv::resize(labeled, labeled, cv::Size(), scale, scale);
    processed.push_back(labeled);
  }
  size_t rows = (processed.size() + cols - 1) / cols;
  std::vector<cv::Mat> rowMats;
  size_t idx = 0;
  for (size_t r = 0; r < rows; ++r) {
    std::vector<cv::Mat> rowImgs;
    for (int c = 0; c < cols && idx < processed.size(); ++c, ++idx) {
      rowImgs.push_back(processed[idx]);
    }
    cv::Mat row;
    cv::hconcat(rowImgs, row);
    rowMats.push_back(row);
  }
  cv::Mat mosaic;
  cv::vconcat(rowMats, mosaic);
  return mosaic;
}

// Update the onTrackbar callback to render into single window
static void onTrackbar(int, void*) {
  // sync special slider values back to params
  g_detector.setParams(g_params);
  // Update main detector as well so mosaic in main loop matches
  // (g_detector is only used inside this callback)
  // Note: we assume detector reference will be updated externally when exiting pause
  // but for safety we sync here if globals available.
  std::vector<cv::Mat> inters;
  cv::Mat dbg = g_detector.process(g_frozen, &inters);
  std::vector<std::pair<cv::Mat,std::string>> tiles;
  tiles.push_back({dbg, "debug"});
  const std::vector<std::string> lbls = {"mask","opened","blurred","contours","debug"};
  for(size_t i=0;i<inters.size() && i<lbls.size();++i){
    tiles.push_back({inters[i], lbls[i]});
  }
  cv::Mat mosaic = buildMosaic(tiles);
  cv::Mat disp = fitToWindow(mosaic);
  if (!disp.empty())
    cv::imshow(WINDOW_NAME, disp);
}

int main() {
  HikCamera camera;
  if (!camera.open()) {
    std::cerr << "Failed to open Hik camera" << std::endl;
    return -1;
  }

  HoopDetector detector;
  g_detector = detector; // copy (initial)
  g_params = detector.params();

  bool recording = false;
  cv::VideoWriter writer;
  bool paused = false;

  cv::namedWindow(WINDOW_NAME, cv::WINDOW_NORMAL);
  cv::resizeWindow(WINDOW_NAME, DISP_W, DISP_H);

  while (true) {
    cv::Mat frame;
    if (!paused) {
      if (!camera.getFrame(frame)) {
        std::cerr << "Grab frame failed" << std::endl;
        continue;
      }
      g_frozen = frame.clone(); // keep latest for potential pause
    } else {
      frame = g_frozen.clone();
    }

    std::vector<cv::Mat> inters;
    cv::Mat debug = detector.process(frame, &inters);
    std::vector<std::pair<cv::Mat,std::string>> tiles = {{debug,"debug"}};
    const std::vector<std::string> lbls2 = {"mask","opened","blurred","contours","debug"};
    for(size_t i=0;i<inters.size() && i<lbls2.size();++i){
      tiles.push_back({inters[i], lbls2[i]});
    }
    cv::Mat mosaic = buildMosaic(tiles);
    cv::Mat disp = fitToWindow(mosaic);
    if (!disp.empty())
      cv::imshow(WINDOW_NAME, disp);

    // Write to video if recording
    if (recording && writer.isOpened() && !paused) {
      writer.write(frame);
    }

    char key = (char)cv::waitKey(1);
    if (key == 'q' || key==27) { // q or ESC to quit
      break;
    }

    if (key == 'v') { // start/stop video recording
      if (!recording) {
        std::string filename = "record_" + timestampString() + ".avi";
        int fourcc = cv::VideoWriter::fourcc('M', 'J', 'P', 'G');
        writer.open(filename, fourcc, 30.0, frame.size());
        if (!writer.isOpened()) {
          std::cerr << "Failed to open VideoWriter" << std::endl;
        } else {
          std::cout << "Start recording to " << filename << std::endl;
          recording = true;
        }
      } else {
        recording = false;
        writer.release();
        std::cout << "Stop recording" << std::endl;
      }
    } else if (key == 'p') {
      paused = !paused;
      if (paused) {
        // Enter pause mode: show trackbars ------------------------------------------------
        cv::namedWindow("control", cv::WINDOW_NORMAL);
        cv::createTrackbar("morph_kernel", "control", &g_params.morph_kernel, 31, onTrackbar);
        cv::createTrackbar("median_kernel", "control", &g_params.median_kernel, 31, onTrackbar);
        // Initial call will draw
        onTrackbar(0,nullptr);
      } else {
        cv::destroyWindow("control");
        detector.setParams(g_params);
        g_detector.setParams(g_params);
      }
    }
  }

  if (writer.isOpened()) writer.release();

  camera.close();
  cv::destroyAllWindows();
  return 0;
} 