#include "hik_camera.hpp"

#include <opencv2/opencv.hpp>
#include <iostream>
#include <limits>
#include <vector>
#include <algorithm>
#include <ctime>
#include <sstream>
#include <filesystem>
#include <string>

#include <getopt.h>

using namespace non_ros_hoop;

struct HSVRange {
  int h_min = 179, s_min = 255, v_min = 255;
  int h_max = 0,   s_max = 0,   v_max = 0;

  void update(const cv::Vec3b &hsv){
    h_min = std::min(h_min, (int)hsv[0]);
    s_min = std::min(s_min, (int)hsv[1]);
    v_min = std::min(v_min, (int)hsv[2]);
    h_max = std::max(h_max, (int)hsv[0]);
    s_max = std::max(s_max, (int)hsv[1]);
    v_max = std::max(v_max, (int)hsv[2]);
  }

  cv::Scalar lower() const { return {h_min,s_min,v_min}; }
  cv::Scalar upper() const { return {h_max,s_max,v_max}; }
};

// -----------------------------------------------------------------------------
// Config
// -----------------------------------------------------------------------------
static constexpr int DISPLAY_WIDTH = 1280;  // fixed window width in pixels

static HSVRange g_range;
static cv::Mat g_current;
static std::vector<cv::Point> g_click_pts;
static double g_scale = 1.0;  // mosaic -> display scale factor (set every frame)

// -----------------------------------------------------------------------------
// CLI argument parsing
// -----------------------------------------------------------------------------
static std::string g_image_path;

static void parseArgs(int argc, char** argv){
  const char* const short_opts = "i:";
  const option long_opts[] = {
    {"image", required_argument, nullptr, 'i'},
    {nullptr, 0, nullptr, 0}
  };

  while(true){
    const auto opt = getopt_long(argc, argv, short_opts, long_opts, nullptr);
    if(opt == -1) break;
    switch(opt){
      case 'i': g_image_path = optarg; break;
      default: break;
    }
  }
}

void onMouse(int event, int x, int y, int /*flags*/, void*){
  if(event == cv::EVENT_LBUTTONDOWN && !g_current.empty()){
    // Reverse-scale the click coordinates to mosaic coordinates
    int mx = static_cast<int>(x * g_scale + 0.5);
    int my = static_cast<int>(y * g_scale + 0.5);

    // Only process clicks inside the left (original) image
    if(mx < g_current.cols && my < g_current.rows){
      cv::Mat hsv;
      cv::cvtColor(g_current, hsv, cv::COLOR_BGR2HSV);
      cv::Vec3b pix = hsv.at<cv::Vec3b>(my, mx);

      g_range.update(pix);
      g_click_pts.emplace_back(mx, my);

      std::cout << "Clicked HSV: ("<< (int)pix[0] <<","<< (int)pix[1]<<","<< (int)pix[2] <<")\n";
      std::cout << "Current range lower=("<<g_range.h_min<<","<<g_range.s_min<<","<<g_range.v_min
                <<") upper=("<<g_range.h_max<<","<<g_range.s_max<<","<<g_range.v_max<<")\n";
    }
  }
}

int main(int argc, char** argv){
  parseArgs(argc, argv);

  HikCamera cam;
  bool use_camera = g_image_path.empty();
  if(use_camera){
    if(!cam.open()) return -1;
  } else {
    g_current = cv::imread(g_image_path);
    if(g_current.empty()){
      std::cerr << "Failed to load image: "<< g_image_path <<"\n";
      return -1;
    }
    std::cout << "Loaded image "<< g_image_path <<" for calibration. Press 'q' to quit, 'c' to clear."<< std::endl;
  }

  const std::string win="hsv_calib";
  cv::namedWindow(win, cv::WINDOW_NORMAL);
  cv::setMouseCallback(win, onMouse);

  bool paused=false;

  while(true){
    if(use_camera && !paused){
      if(!cam.getFrame(g_current)) continue;
    }

    cv::Mat hsv, mask;
    if(g_range.h_min<=g_range.h_max)
    {
      cv::cvtColor(g_current, hsv, cv::COLOR_BGR2HSV);
      cv::inRange(hsv, g_range.lower(), g_range.upper(), mask);
    }
    else mask = cv::Mat::zeros(g_current.size(), CV_8UC1);

    // Draw click points in red on a copy of current frame
    cv::Mat curr_with_marks = g_current.clone();
    for(const auto& pt : g_click_pts){
      cv::circle(curr_with_marks, pt, 4, cv::Scalar(0,0,255), -1);
    }

    // Compose mosaic (original + mask)
    cv::Mat mask_bgr;
    cv::cvtColor(mask, mask_bgr, cv::COLOR_GRAY2BGR);
    cv::Mat mosaic;
    cv::hconcat(curr_with_marks, mask_bgr, mosaic);

    // Resize to fixed display width while preserving aspect ratio
    g_scale = static_cast<double>(mosaic.cols) / DISPLAY_WIDTH;
    cv::Mat mosaic_disp;
    if(g_scale > 1.0){
      int disp_height = static_cast<int>(mosaic.rows / g_scale);
      cv::resize(mosaic, mosaic_disp, cv::Size(DISPLAY_WIDTH, disp_height));
    } else {
      g_scale = 1.0;  // no scaling
      mosaic_disp = mosaic;
    }

    cv::imshow(win, mosaic_disp);

    char k = (char)cv::waitKey(30);
    if(k=='q' || k==27) break;
    if(k=='p') paused=!paused;
    if(k=='s' && paused){
      // Save current source frame (without mosaic scaling) to PNG
      std::time_t t = std::time(nullptr);
      std::stringstream ss;
      ss << "snapshot_" << t << ".png";
      cv::imwrite(ss.str(), g_current);
      std::cout << "Saved snapshot to "<< ss.str() <<"\n";
    }
    if(k=='c') {
        // clear current range
        g_range = HSVRange();
        g_click_pts.clear();
        std::cout << "Cleared collected HSV samples.\n";
    }
  }

  std::cout << "\nFinal HSV threshold:\n";
  std::cout << "lower: ["<<g_range.h_min<<","<<g_range.s_min<<","<<g_range.v_min<<"]\n";
  std::cout << "upper: ["<<g_range.h_max<<","<<g_range.s_max<<","<<g_range.v_max<<"]\n";

  cam.close();
  return 0;
} 