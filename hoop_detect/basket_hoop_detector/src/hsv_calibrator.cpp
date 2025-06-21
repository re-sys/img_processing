#include "basket_hoop_detector/hik_camera.hpp"

#include <opencv2/opencv.hpp>
#include <yaml-cpp/yaml.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <filesystem>

using basket_hoop_detector::HikCamera;

class HSVCalibrator {
public:
  HSVCalibrator(const std::string &output_file)
    : output_file_(output_file) {}

  bool run() {
    if(!camera_.open()) {
      std::cerr << "[HSVCalibrator] Failed to open camera" << std::endl;
      return false;
    }

    std::cout << "---------------------------------------------\n";
    std::cout << "HSV Calibrator Usage:\n";
    std::cout << " - Left click : sample pixel HSV\n";
    std::cout << " - 's' key    : save YAML & exit\n";
    std::cout << " - 'q' key    : quit without saving\n";
    // ensure directory for yaml exists
    std::filesystem::path out_path(output_file_);
    if(!out_path.parent_path().empty())
        std::filesystem::create_directories(out_path.parent_path());

    std::cout << "Output file   : " << output_file_ << "\n";
    std::cout << "---------------------------------------------" << std::endl;

    cv::namedWindow("calibrator");
    cv::setMouseCallback("calibrator", onMouseStatic, this);

    while(true){
      cv::Mat frame;
      if(!camera_.getFrame(frame)) continue;
      latest_frame_ = frame.clone();

      // optional overlay: show current HSV range
      if(has_range_){
        cv::putText(frame, rangeText(), {10,30}, cv::FONT_HERSHEY_SIMPLEX, 0.7, {0,255,0},2);
      }
      cv::imshow("calibrator", frame);
      char key = (char)cv::waitKey(1);
      if(key=='q') break;
      if(key=='s'){
        if(!has_range_){
          std::cerr << "[HSVCalibrator] No samples collected, cannot save!" << std::endl;
          continue;
        }
        saveYaml();
        std::cout << "[HSVCalibrator] Saved config to " << output_file_ << std::endl;
        break;
      }
    }
    camera_.close();
    cv::destroyAllWindows();
    return true;
  }

private:
  static void onMouseStatic(int event,int x,int y,int flags,void* userdata){
    reinterpret_cast<HSVCalibrator*>(userdata)->onMouse(event,x,y,flags);
  }

  void onMouse(int event,int x,int y,int /*flags*/){
    if(event!=cv::EVENT_LBUTTONDOWN) return;
    if(latest_frame_.empty()) return;
    cv::Mat hsv;
    cv::cvtColor(latest_frame_, hsv, cv::COLOR_BGR2HSV);
    cv::Vec3b pix = hsv.at<cv::Vec3b>(y,x);
    samples_.push_back(pix);
    updateRange();
    std::cout << "Sample " << samples_.size() << ": H=" << (int)pix[0]
              << " S=" << (int)pix[1] << " V=" << (int)pix[2]
              << " => range [" << hsv_lower_[0] << "," << hsv_lower_[1] << "," << hsv_lower_[2]
              << "]- [" << hsv_upper_[0] << "," << hsv_upper_[1] << "," << hsv_upper_[2] << "]\n";
  }

  void updateRange(){
    if(samples_.empty()) return;
    int h_min=180,s_min=255,v_min=255;
    int h_max=0,s_max=0,v_max=0;
    for(auto &p:samples_){
      h_min=std::min(h_min,(int)p[0]);
      s_min=std::min(s_min,(int)p[1]);
      v_min=std::min(v_min,(int)p[2]);
      h_max=std::max(h_max,(int)p[0]);
      s_max=std::max(s_max,(int)p[1]);
      v_max=std::max(v_max,(int)p[2]);
    }
    int pad=10;
    hsv_lower_=cv::Scalar(std::max(0,h_min-pad), std::max(0,s_min-pad), std::max(0,v_min-pad));
    hsv_upper_=cv::Scalar(std::min(179,h_max+pad), std::min(255,s_max+pad), std::min(255,v_max+pad));
    has_range_=true;
  }

  std::string rangeText(){
    char buf[128];
    std::snprintf(buf,sizeof(buf),"L[%d,%d,%d] U[%d,%d,%d]", (int)hsv_lower_[0],(int)hsv_lower_[1],(int)hsv_lower_[2], (int)hsv_upper_[0],(int)hsv_upper_[1],(int)hsv_upper_[2]);
    return std::string(buf);
  }

  void saveYaml(){
    YAML::Node root;
    root["hsv"]["lower"].push_back((int)hsv_lower_[0]);
    root["hsv"]["lower"].push_back((int)hsv_lower_[1]);
    root["hsv"]["lower"].push_back((int)hsv_lower_[2]);
    root["hsv"]["upper"].push_back((int)hsv_upper_[0]);
    root["hsv"]["upper"].push_back((int)hsv_upper_[1]);
    root["hsv"]["upper"].push_back((int)hsv_upper_[2]);

    // default RANSAC params
    root["ransac"]["iterations"] = 200;
    root["ransac"]["tolerance"]  = 0.05;

    std::ofstream fout(output_file_);
    fout << root;
  }

  // members
  HikCamera camera_;
  std::string output_file_;
  cv::Mat latest_frame_;
  std::vector<cv::Vec3b> samples_;
  cv::Scalar hsv_lower_{0,0,0}, hsv_upper_{179,255,255};
  bool has_range_ = false;
};

int main(int argc,char** argv){
  std::string out_file="config/hsv_config.yaml";
  if(argc>1) out_file=argv[1];
  HSVCalibrator calib(out_file);
  calib.run();
  return 0;
} 