#include "hik_camera.hpp"
#include "hoop_detector.hpp"

#include <opencv2/opencv.hpp>
#include <chrono>
#include <filesystem>
#include <iomanip>
#include <sstream>
#include <vector>

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

// Utility: build a tiled mosaic from multiple images (2 columns)
static cv::Mat buildMosaic(const std::vector<cv::Mat> &imgs, int cols = 2) {
  if (imgs.empty()) return cv::Mat();
  std::vector<cv::Mat> processed;
  const int target_h = 300; // resize height for each tile
  for (const auto &im : imgs) {
    cv::Mat colored;
    if (im.channels() == 1)
      cv::cvtColor(im, colored, cv::COLOR_GRAY2BGR);
    else
      colored = im;
    double scale = static_cast<double>(target_h) / colored.rows;
    cv::resize(colored, colored, cv::Size(), scale, scale);
    processed.push_back(colored);
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
  g_detector.setParams(g_params);
  std::vector<cv::Mat> inters;
  cv::Mat dbg = g_detector.process(g_frozen, &inters);
  std::vector<cv::Mat> tiles;
  tiles.push_back(dbg);
  tiles.insert(tiles.end(), inters.begin(), inters.end());
  cv::Mat mosaic = buildMosaic(tiles);
  if (!mosaic.empty())
    cv::imshow(WINDOW_NAME, mosaic);
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

  cv::namedWindow(WINDOW_NAME, cv::WINDOW_NORMAL);

  while (true) {
    cv::Mat frame;
    if (!camera.getFrame(frame)) {
      std::cerr << "Grab frame failed" << std::endl;
      continue;
    }

    cv::Mat debug = detector.process(frame);
    cv::Mat mosaic = buildMosaic({debug});
    if (!mosaic.empty())
      cv::imshow(WINDOW_NAME, mosaic);

    // Write to video if recording
    if (recording && writer.isOpened()) {
      writer.write(frame);
    }

    char key = (char)cv::waitKey(1);
    if (key == 27) { // ESC to quit
      break;
    }

    if (key == 'r') {
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
      }
    } else if (key == 'q') {
      if (recording) {
        recording = false;
        writer.release();
        std::cout << "Stop recording" << std::endl;
      }
    } else if (key == 'c') {
      // Enter frozen debug mode ---------------------------------------------
      g_frozen = frame.clone();

      // Ensure display window exists (recreate to clear previous content)
      cv::namedWindow(WINDOW_NAME, cv::WINDOW_NORMAL);
      // Create control panel window with trackbars
      cv::namedWindow("control", cv::WINDOW_NORMAL);
      // Attach trackbars to g_params
      cv::createTrackbar("morph_kernel", "control", &g_params.morph_kernel, 31, onTrackbar);
      cv::createTrackbar("median_kernel", "control", &g_params.median_kernel, 31, onTrackbar);
      cv::createTrackbar("h_lower", "control", &g_params.h_lower, 179, onTrackbar);
      cv::createTrackbar("h_upper", "control", &g_params.h_upper, 179, onTrackbar);
      cv::createTrackbar("s_lower", "control", &g_params.s_lower, 255, onTrackbar);
      cv::createTrackbar("s_upper", "control", &g_params.s_upper, 255, onTrackbar);
      cv::createTrackbar("v_lower", "control", &g_params.v_lower, 255, onTrackbar);
      cv::createTrackbar("v_upper", "control", &g_params.v_upper, 255, onTrackbar);

      // For area and other double params we will scale to int trackbars
      int min_area_tb = static_cast<int>(g_params.min_area);
      int max_area_tb = static_cast<int>(g_params.max_area);
      int ransac_tb  = static_cast<int>(g_params.ransac_thresh);
      cv::createTrackbar("min_area", "control", &min_area_tb, 100000, onTrackbar);
      cv::createTrackbar("max_area", "control", &max_area_tb, 2000000, onTrackbar);
      cv::createTrackbar("ransac_thresh", "control", &ransac_tb, 100, onTrackbar);

      // Initial call
      onTrackbar(0, nullptr);

      // Loop inside debug mode
      while (true) {
        char k = (char)cv::waitKey(30);
        if (k == 'q') {
          break; // exit debug mode
        }
        // Update params from trackbars
        g_params.min_area = std::max(1, min_area_tb);
        g_params.max_area = std::max(g_params.min_area + 1, max_area_tb);
        g_params.ransac_thresh = std::max(1, ransac_tb);
        detector.setParams(g_params);
        g_detector.setParams(g_params);
      }

      cv::destroyWindow("control");
      // clear the display of debug tiles back to live view
      cv::Mat dbg_only = buildMosaic({detector.process(frame)});
      if (!dbg_only.empty())
        cv::imshow(WINDOW_NAME, dbg_only);
    }
  }

  if (writer.isOpened()) writer.release();

  camera.close();
  cv::destroyAllWindows();
  return 0;
} 