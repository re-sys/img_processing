#include "basket_hoop_detector/hik_camera.hpp"

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <opencv2/opencv.hpp>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.hpp>
#include <tf2_ros/transform_broadcaster.h>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <yaml-cpp/yaml.h>

#include <random>
#include <chrono>

using namespace std::chrono_literals;
using basket_hoop_detector::HikCamera;

class HoopDetectorNode : public rclcpp::Node {
public:
  HoopDetectorNode() : Node("hoop_detector_node") {
    declare_parameter<int>("debug_level", 0);
    declare_parameter<double>("hoop_radius", 0.23); // 23 cm default
    declare_parameter<std::vector<double>>("camera_matrix", {600,0,320, 0,600,240, 0,0,1});
    declare_parameter<std::vector<double>>("dist_coeffs", {0,0,0,0,0});
    declare_parameter<std::string>("config_file", "config/hsv_config.yaml");
    declare_parameter<std::vector<int>>("hsv_lower", {});
    declare_parameter<std::vector<int>>("hsv_upper", {});
    declare_parameter<int>("ransac_iterations_param", -1);
    declare_parameter<double>("ransac_tolerance_param", -1.0);

    debug_level_ = get_parameter("debug_level").as_int();
    hoop_radius_ = get_parameter("hoop_radius").as_double();

    auto cm = get_parameter("camera_matrix").as_double_array();
    auto dc = get_parameter("dist_coeffs").as_double_array();

    camera_matrix_ = (cv::Mat_<double>(3,3) << cm[0],cm[1],cm[2], cm[3],cm[4],cm[5], cm[6],cm[7],cm[8]);
    dist_coeffs_ = cv::Mat(dc.size(),1,CV_64F);
    for(size_t i=0;i<dc.size();++i) dist_coeffs_.at<double>(i)=dc[i];

    // Load HSV & RANSAC config
    auto cfg_file = get_parameter("config_file").as_string();
    if(!loadConfig(cfg_file)){
      RCLCPP_ERROR(get_logger(), "Failed to load config file %s", cfg_file.c_str());
      rclcpp::shutdown();
      return;
    }

    // Override with ROS parameters if provided
    auto lower_vec = get_parameter("hsv_lower").as_integer_array();
    auto upper_vec = get_parameter("hsv_upper").as_integer_array();
    if(lower_vec.size()==3 && upper_vec.size()==3){
      hsv_lower_ = cv::Scalar(lower_vec[0],lower_vec[1],lower_vec[2]);
      hsv_upper_ = cv::Scalar(upper_vec[0],upper_vec[1],upper_vec[2]);
      if(debug_level_>0)
        RCLCPP_INFO(get_logger(), "[Param] Override HSV lower=(%ld,%ld,%ld) upper=(%ld,%ld,%ld)", lower_vec[0],lower_vec[1],lower_vec[2], upper_vec[0],upper_vec[1],upper_vec[2]);
    }

    int r_i = get_parameter("ransac_iterations_param").as_int();
    double r_tol = get_parameter("ransac_tolerance_param").as_double();
    if(r_i>0) ransac_iterations_ = r_i;
    if(r_tol>0) ransac_tol_ = r_tol;

    if(debug_level_>0){
      RCLCPP_INFO(get_logger(), "Final HSV lower=(%d,%d,%d) upper=(%d,%d,%d)", (int)hsv_lower_[0],(int)hsv_lower_[1],(int)hsv_lower_[2], (int)hsv_upper_[0],(int)hsv_upper_[1],(int)hsv_upper_[2]);
      RCLCPP_INFO(get_logger(), "Final RANSAC iterations=%d tol=%.3f", ransac_iterations_, ransac_tol_);
    }

    if(!camera_.open()) {
      RCLCPP_ERROR(get_logger(), "Failed to open Hikvision camera");
      rclcpp::shutdown();
      return;
    }

    // Create image publisher for debug
    img_pub_ = image_transport::create_publisher(this, "debug_image");

    // Timer to run at ~30Hz
    timer_ = create_wall_timer(33ms, std::bind(&HoopDetectorNode::processFrame, this));

    tf_broadcaster_ = std::make_shared<tf2_ros::TransformBroadcaster>(*this);
  }

  ~HoopDetectorNode() {
    camera_.close();
    cv::destroyAllWindows();
  }

private:
  bool loadConfig(const std::string &file){
    try{
      YAML::Node root = YAML::LoadFile(file);
      auto lower = root["hsv"]["lower"];
      auto upper = root["hsv"]["upper"];
      if(!lower || !upper || lower.size()!=3 || upper.size()!=3){
        RCLCPP_ERROR(get_logger(), "Invalid HSV config structure");
        return false;
      }
      hsv_lower_ = cv::Scalar(lower[0].as<int>(), lower[1].as<int>(), lower[2].as<int>());
      hsv_upper_ = cv::Scalar(upper[0].as<int>(), upper[1].as<int>(), upper[2].as<int>());

      auto ransac = root["ransac"];
      if(ransac){
        ransac_iterations_ = ransac["iterations"].as<int>(200);
        ransac_tol_ = ransac["tolerance"].as<double>(0.05);
      }
      else{
        ransac_iterations_ = 200;
        ransac_tol_ = 0.05;
      }

      // camera intrinsics
      auto cam = root["camera"];
      if(cam && cam["matrix"] && cam["dist"]){
        auto mat = cam["matrix"];
        auto dist = cam["dist"];
        if(mat.size()==9){
          camera_matrix_ = (cv::Mat_<double>(3,3) << mat[0].as<double>(), mat[1].as<double>(), mat[2].as<double>(),
                                                            mat[3].as<double>(), mat[4].as<double>(), mat[5].as<double>(),
                                                            mat[6].as<double>(), mat[7].as<double>(), mat[8].as<double>());
        }
        dist_coeffs_ = cv::Mat(dist.size(),1,CV_64F);
        for(size_t i=0;i<dist.size();++i) dist_coeffs_.at<double>(i)=dist[i].as<double>();
      }

      RCLCPP_INFO(get_logger(), "Loaded HSV lower=(%d,%d,%d) upper=(%d,%d,%d)", (int)hsv_lower_[0],(int)hsv_lower_[1],(int)hsv_lower_[2], (int)hsv_upper_[0],(int)hsv_upper_[1],(int)hsv_upper_[2]);
      RCLCPP_INFO(get_logger(), "RANSAC iterations=%d tol=%.3f", ransac_iterations_, ransac_tol_);
      return true;
    }catch(const std::exception &e){
      RCLCPP_ERROR(get_logger(), "YAML exception: %s", e.what());
      return false;
    }
  }

  void processFrame(){
    cv::Mat frame;
    if(!camera_.getFrame(frame)) return;
    latest_frame_ = frame.clone();

    // Show selector window irrespective of calibration status
    cv::imshow("preview", frame);
    cv::waitKey(1);

    cv::Mat hsv,mask;
    cv::cvtColor(frame, hsv, cv::COLOR_BGR2HSV);
    cv::inRange(hsv, hsv_lower_, hsv_upper_, mask);

    // Morphology
    cv::erode(mask, mask, cv::Mat(), cv::Point(-1,-1), 2);
    cv::dilate(mask, mask, cv::Mat(), cv::Point(-1,-1), 2);
    cv::medianBlur(mask, mask, 5);

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
    std::vector<cv::Point> all_pts;
    for(const auto &c:contours) {
      all_pts.insert(all_pts.end(), c.begin(), c.end());
    }

    cv::Point2f best_center;
    float best_radius=0;
    size_t best_inliers=0;

    if(all_pts.size() >= 3){
      std::mt19937 rng(std::random_device{}());
      std::uniform_int_distribution<size_t> dist(0, all_pts.size()-1);
      for(int iter=0; iter<ransac_iterations_; ++iter){
        // random 3 distinct points
        size_t i1 = dist(rng), i2 = dist(rng), i3 = dist(rng);
        if(i1==i2 || i1==i3 || i2==i3) {--iter; continue;}
        cv::Point2f p1=all_pts[i1], p2=all_pts[i2], p3=all_pts[i3];
        // circle from 3 points
        float denom = 2*(p1.x*(p2.y-p3.y)+p2.x*(p3.y-p1.y)+p3.x*(p1.y-p2.y));
        if(fabs(denom) < 1e-6) continue;
        float A = p1.dot(p1), B = p2.dot(p2), C = p3.dot(p3);
        float cx = (A*(p2.y-p3.y)+B*(p3.y-p1.y)+C*(p1.y-p2.y))/denom;
        float cy = (A*(p3.x-p2.x)+B*(p1.x-p3.x)+C*(p2.x-p1.x))/denom;
        cv::Point2f center(cx,cy);
        float radius = cv::norm(center-p1);
        // count inliers
        size_t inliers=0;
        for(const auto &pt: all_pts){
          float d = fabs(cv::norm(center-pt)-radius);
          if(d < ransac_tol_*radius) ++inliers;
        }
        if(inliers>best_inliers){
          best_inliers=inliers;
          best_center=center;
          best_radius=radius;
        }
      }
    }

    cv::Mat debug_img=frame.clone();
    if(best_inliers>0){
      cv::circle(debug_img, best_center, (int)best_radius, {0,255,0},2);
      // PnP
      std::vector<cv::Point2f> img_pts = {
        {best_center.x+best_radius, best_center.y},
        {best_center.x-best_radius, best_center.y},
        {best_center.x, best_center.y+best_radius},
        {best_center.x, best_center.y-best_radius}
      };
      std::vector<cv::Point3f> obj_pts = {
        {hoop_radius_,0,0}, {-hoop_radius_,0,0}, {0,hoop_radius_,0}, {0,-hoop_radius_,0}
      };
      cv::Mat rvec,tvec;
      bool pnp_ok = cv::solvePnP(obj_pts, img_pts, camera_matrix_, dist_coeffs_, rvec, tvec, false, cv::SOLVEPNP_ITERATIVE);
      if(pnp_ok){
        geometry_msgs::msg::TransformStamped tf_msg;
        tf_msg.header.stamp = this->now();
        tf_msg.header.frame_id = "camera_link";
        tf_msg.child_frame_id = "basket_hoop";
        cv::Mat R;
        cv::Rodrigues(rvec,R);
        // Convert rotation matrix to quaternion
        Eigen::Matrix3d eR;
        for(int i=0;i<3;++i)
          for(int j=0;j<3;++j)
            eR(i,j)=R.at<double>(i,j);
        Eigen::Quaterniond q(eR);
        tf_msg.transform.translation.x = tvec.at<double>(0);
        tf_msg.transform.translation.y = tvec.at<double>(1);
        tf_msg.transform.translation.z = tvec.at<double>(2);
        tf_msg.transform.rotation.x = q.x();
        tf_msg.transform.rotation.y = q.y();
        tf_msg.transform.rotation.z = q.z();
        tf_msg.transform.rotation.w = q.w();
        tf_broadcaster_->sendTransform(tf_msg);
      }
    }

    if(debug_level_>0){
      cv::imshow("mask", mask);
      cv::imshow("debug", debug_img);
    }

    // Publish debug image
    if(img_pub_.getNumSubscribers()>0){
      sensor_msgs::msg::Image::SharedPtr ros_img = cv_bridge::CvImage(std_msgs::msg::Header(), "bgr8", debug_img).toImageMsg();
      img_pub_.publish(ros_img);
    }
  }

  // Members
  HikCamera camera_;
  rclcpp::TimerBase::SharedPtr timer_;
  image_transport::Publisher img_pub_;
  std::shared_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;

  // HSV threshold from YAML
  cv::Scalar hsv_lower_{0,0,0}, hsv_upper_{179,255,255};

  // RANSAC parameters
  int ransac_iterations_ = 200;
  double ransac_tol_ = 0.05;

  cv::Mat latest_frame_;

  // Parameters
  int debug_level_ = 0;
  double hoop_radius_ = 0.23;
  cv::Mat camera_matrix_, dist_coeffs_;
};

int main(int argc, char **argv){
  rclcpp::init(argc, argv);
  auto node = std::make_shared<HoopDetectorNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
} 