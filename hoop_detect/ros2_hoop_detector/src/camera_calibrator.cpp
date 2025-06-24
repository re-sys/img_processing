#include "basket_hoop_detector/hik_camera.hpp"

#include <opencv2/opencv.hpp>
#include <yaml-cpp/yaml.h>
#include <filesystem>
#include <iostream>
#include <vector>
#include <string>

namespace fs = std::filesystem;
using basket_hoop_detector::HikCamera;

class CameraCalibrator {
public:
  CameraCalibrator(const std::string &picture_dir,
                   const std::string &output_yaml,
                   cv::Size board_size = {11,8},
                   float square_size = 0.02f)  // 2.5 cm
    : picture_dir_(picture_dir),
      output_yaml_(output_yaml),
      board_size_(board_size),
      square_size_(square_size) {}

  bool run() {
    if(!fs::exists(picture_dir_)) fs::create_directories(picture_dir_);

    if(!camera_.open()) {
      std::cerr << "[CameraCalibrator] Failed to open camera" << std::endl;
      return false;
    }

    std::cout << "-----------------------------------------\n";
    std::cout << "Camera Calibrator Instructions\n";
    std::cout << "  's' : save current frame to pictures/\n";
    std::cout << "  'c' : start calibration using saved images\n";
    std::cout << "  'q' : quit\n";
    std::cout << "Saved images directory: " << picture_dir_ << "\n";
    std::cout << "Output YAML: " << output_yaml_ << "\n";
    std::cout << "Chessboard: " << board_size_.width << "x" << board_size_.height
              << " squares, square_size=" << square_size_ << " m\n";
    std::cout << "-----------------------------------------" << std::endl;

    cv::namedWindow("calib_view");

    while(true){
      cv::Mat frame;
      if(!camera_.getFrame(frame)) continue;

      cv::imshow("calib_view", frame);
      char key = (char)cv::waitKey(1);
      if(key=='q') break;
      if(key=='s'){
        saveImage(frame);
      }
      if(key=='c'){
        calibrate();
      }
    }
    camera_.close();
    cv::destroyAllWindows();
    return true;
  }

private:
  void saveImage(const cv::Mat &img){
    std::string filename = picture_dir_ + "/img_" + std::to_string(counter_++) + ".png";
    cv::imwrite(filename, img);
    std::cout << "[CameraCalibrator] Saved " << filename << std::endl;
  }

  void calibrate(){
    std::vector<std::string> image_files;
    for(auto &p: fs::directory_iterator(picture_dir_)){
      if(p.is_regular_file()){
        std::string ext = p.path().extension().string();
        if(ext==".png" || ext==".jpg" || ext==".jpeg")
          image_files.push_back(p.path().string());
      }
    }
    if(image_files.size()<5){
      std::cerr << "[CameraCalibrator] Need at least 5 images, currently " << image_files.size() << std::endl;
      return;
    }

    std::vector<std::vector<cv::Point3f>> object_points;
    std::vector<std::vector<cv::Point2f>> image_points;

    // prepare object points template
    std::vector<cv::Point3f> objp;
    for(int i=0;i<board_size_.height;++i)
      for(int j=0;j<board_size_.width;++j)
        objp.emplace_back(j*square_size_, i*square_size_, 0);

    for(const auto &file: image_files){
      cv::Mat img = cv::imread(file);
      if(img.empty()) continue;
      cv::Mat gray;
      cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
      std::vector<cv::Point2f> corners;
      bool found = cv::findChessboardCorners(gray, board_size_, corners,
                                             cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_FAST_CHECK | cv::CALIB_CB_NORMALIZE_IMAGE);
      if(found){
        cv::cornerSubPix(gray, corners, {11,11}, {-1,-1},
                         cv::TermCriteria(cv::TermCriteria::EPS+cv::TermCriteria::MAX_ITER, 30, 0.001));
        image_points.push_back(corners);
        object_points.push_back(objp);
      }
    }
    if(image_points.size()<5){
      std::cerr << "[CameraCalibrator] Not enough valid chessboard detections: " << image_points.size() << std::endl;
      return;
    }

    cv::Mat camera_matrix, dist_coeffs;
    std::vector<cv::Mat> rvecs, tvecs;
    double rms = cv::calibrateCamera(object_points, image_points, board_size_,
                         camera_matrix, dist_coeffs, rvecs, tvecs);
    std::cout << "[CameraCalibrator] Calibration done, RMS error=" << rms << std::endl;
    std::cout << "Camera Matrix:\n" << camera_matrix << std::endl;
    std::cout << "Dist Coeffs: " << dist_coeffs.t() << std::endl;

    // write to YAML (append or create)
    YAML::Node root;
    if(fs::exists(output_yaml_)){
      root = YAML::LoadFile(output_yaml_);
    }
    for(int i=0;i<9;++i) root["camera"]["matrix"].push_back(camera_matrix.at<double>(i/3,i%3));
    for(int i=0;i<dist_coeffs.total();++i) root["camera"]["dist"].push_back(dist_coeffs.at<double>(i));

    std::ofstream fout(output_yaml_);
    fout << root;
    std::cout << "[CameraCalibrator] Wrote intrinsics to " << output_yaml_ << std::endl;
  }

  // members
  HikCamera camera_;
  std::string picture_dir_;
  std::string output_yaml_;
  cv::Size board_size_;
  float square_size_;
  int counter_ = 0;
};

int main(int argc,char** argv){
  std::string pic_dir = "pictures";
  std::string yaml_file = "config/hsv_config.yaml";
  if(argc>1) pic_dir = argv[1];
  if(argc>2) yaml_file = argv[2];
  CameraCalibrator calib(pic_dir, yaml_file);
  calib.run();
  return 0;
} 