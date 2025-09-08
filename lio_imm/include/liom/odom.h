/***********************************************************
 *                                                         *
 * Copyright (c)                                           *
 *                                                         *
 * The Verifiable & Control-Theoretic Robotics (VECTR) Lab *
 * University of California, Los Angeles                   *
 *                                                         *
 * Authors: Kenny J. Chen, Ryan Nemiroff, Brett T. Lopez   *
 * Contact: {kennyjchen, ryguyn, btlopez}@ucla.edu         *
 *                                                         *
 ***********************************************************/

#include "liom/liom.h"

// ROS
#include "rclcpp/rclcpp.hpp"
#include <nav_msgs/msg/odometry.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/pose_array.hpp>
#include <nav_msgs/msg/path.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <tf2_ros/transform_broadcaster.h>
#include <sensor_msgs/msg/image.hpp>  

// BOOST
#include <boost/format.hpp>
#include <boost/circular_buffer.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/range/adaptor/indexed.hpp>
#include <boost/range/adaptor/adjacent_filtered.hpp>

// PCL
#include <pcl/filters/crop_box.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/io/pcd_io.h>
#include <pcl/surface/concave_hull.h>
#include <pcl/surface/convex_hull.h>
#include <pcl_conversions/pcl_conversions.h>

// OpenCV
#include <opencv2/opencv.hpp>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.hpp> 
#include <pcl/surface/gp3.h>
#include <pcl/io/vtk_io.h>
#include <pcl/features/normal_3d.h>  
#include "visualization_msgs/msg/marker.hpp" 

// Optimization
#include <ceres/ceres.h>


class liom::OdomNode: public rclcpp::Node {

public:

  OdomNode();
  ~OdomNode();

  void start();

private:
 
  struct State;
  struct ImuMeas;

  void getParams();

  void callbackPointCloud(const sensor_msgs::msg::PointCloud2::SharedPtr pc);
  void callbackImu(const sensor_msgs::msg::Imu::SharedPtr imu);
  void callbackImage(const sensor_msgs::msg::Image::SharedPtr img);

  void publishPose();

  void publishToROS(pcl::PointCloud<PointType>::ConstPtr published_cloud, Eigen::Matrix4f T_cloud);
  void publishCloud(pcl::PointCloud<PointType>::ConstPtr published_cloud, Eigen::Matrix4f T_cloud);
  void publishKeyframe(std::pair<std::pair<Eigen::Vector3f, Eigen::Quaternionf>,
                       pcl::PointCloud<PointType>::ConstPtr> kf, rclcpp::Time timestamp);

  void getScanFromROS(const sensor_msgs::msg::PointCloud2::SharedPtr& pc);
  void preprocessPoints();
  void deskewPointcloud();
  void initializeInputTarget();
  void setInputSource();

  void initializeDLIO();

  void getNextPose();
  bool imuMeasFromTimeRange(double start_time, double end_time,
                            boost::circular_buffer<ImuMeas>::reverse_iterator& begin_imu_it,
                            boost::circular_buffer<ImuMeas>::reverse_iterator& end_imu_it);
  std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>>
    integrateImu(double start_time, Eigen::Quaternionf q_init, Eigen::Vector3f p_init, Eigen::Vector3f v_init,
                 const std::vector<double>& sorted_timestamps);
  std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>>
    integrateImuInternal(Eigen::Quaternionf q_init, Eigen::Vector3f p_init, Eigen::Vector3f v_init,
                         const std::vector<double>& sorted_timestamps,
                         boost::circular_buffer<ImuMeas>::reverse_iterator begin_imu_it,
                         boost::circular_buffer<ImuMeas>::reverse_iterator end_imu_it);

  std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>>
    integrateImuTest(double start_time, Eigen::Quaternionf q_init, Eigen::Vector3f p_init, Eigen::Vector3f v_init,
                 const std::vector<double>& sorted_timestamps);
  void propagateGICP();

  void propagateState();
  void updateState();

  void setAdaptiveParams();
  void setKeyframeCloud();

  void computeMetrics();
  void computeSpaciousness();
  void computeDensity();

  sensor_msgs::msg::Imu::SharedPtr transformImu(const sensor_msgs::msg::Imu::SharedPtr& imu);

  void updateKeyframes();
  void computeConvexHull();
  void computeConcaveHull();
  void pushSubmapIndices(std::vector<float> dists, int k, std::vector<int> frames);
  void buildSubmap(State vehicle_state);
  void buildKeyframesAndSubmap(State vehicle_state);
  void pauseSubmapBuildIfNeeded();

  void debug();

  rclcpp::TimerBase::SharedPtr publish_timer;

  // Subscribers
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr lidar_sub;
  rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr img_sub;
  rclcpp::CallbackGroup::SharedPtr lidar_cb_group, imu_cb_group, img_cb_group;

  // Publishers
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_pub;
  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr pose_pub;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr path_pub;
  rclcpp::Publisher<geometry_msgs::msg::PoseArray>::SharedPtr kf_pose_pub;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr kf_cloud_pub;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr deskewed_pub;

  rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr mesh_pub;

  /* Modification for 2D-3D */
  image_transport::Publisher pubImage, pubRgb, pubOptFlow;
  std::shared_ptr<image_transport::ImageTransport> it_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr rgb_pub;

  Eigen::Matrix3f Rcl, Rcw;                 // Rotation camera to LiDAR, camera to world
  Eigen::Vector3f Pcl, Pcw;

  // std::atomic<bool> 
  double last_timestamp_img;
  std::mutex mtx_img;
  std::mutex mtx_cp;
  std::mutex mtx_T_corr;

  std::vector<double> timestampsTest;
  bool mahalanobis_idx_;

  cv::Mat getImageFromMsg(const sensor_msgs::msg::Image::ConstSharedPtr &img_msg);
  std::deque<cv::Mat> img_buffer_;
  std::deque<std::vector<cv::Point2f>> pts_buffer;
  using Cloud       =  pcl::PointCloud<PointType>;
  using CloudPtr    =  Cloud::Ptr;
  using CloudRgb    =  pcl::PointCloud<pcl::PointXYZRGB>;
  using CloudRgbPtr =  CloudRgb::Ptr;
  std::deque<CloudPtr> cloud_buffer;
  std::deque<CloudRgbPtr> cloud_rgb_buffer;
  std::deque<double> img_time_buffer_; 
  std::deque<double> cloud_time_buffer;
  std::deque<Eigen::Matrix4f> T_corr_buffer;
  Eigen::Matrix4f T_rgb_corr;
  cv::Mat img_cp;
  double lid_num;
  std::deque<double> lid_num_buffer;
  // void projectLidarToImage(const sensor_msgs::msg::Image::ConstSharedPtr &img);
  void projectLidarToImage(const cv::Mat &img);
  bool visual_initialization, valid_point_flag;

  Eigen::Vector3f accel_total, accel_avg, angl_total, angl_avg;
  double num_imu;
  Eigen::Matrix3f R;

  pcl::PointCloud<pcl::PointXYZRGB>::Ptr valid_colorized_points, prev_colorized_points;
  pcl::PointCloud<PointType>::Ptr rgb_deskewed_scan;  
   
  cv::Mat prev_frame, prev_opt_frame;
  std::vector<cv::Point2f> prev_pts;
  image_transport::Publisher image_pub_;  
  std::vector<cv::Point2f> projected_pts, prev_projected_pts;
  void greedyMatching(const std::vector<cv::Point2f> &projected_pts,
                      const std::vector<cv::Point2f> &featured_pts,
                      float max_dist_thres,
                      std::vector<cv::DMatch> &matches);

  std::vector<cv::DMatch> matches, prev_matches;
  void findCorrespondences(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr &source_cloud,
                           const pcl::PointCloud<pcl::PointXYZRGB>::Ptr &target_cloud,
                           std::vector<int> &correspondences,
                           const std::vector<int> &idxs,
                           const std::vector<int> &dist);

  void calcFlowEstCost(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr &src_cloud,
                       const pcl::PointCloud<pcl::PointXYZRGB>::Ptr &tgt_cloud,
                       const std::vector<cv::Point2f> &curr_pts,
                       const std::vector<cv::Point2f> &prev_pts,
                       const std::vector<int> &frame_idxs);

  void calcRGB(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr &source_cloud,
               const pcl::PointCloud<pcl::PointXYZRGB>::Ptr &target_cloud,
               std::vector<int> &correspondences);

  pcl::PolygonMeshPtr convertToMesh(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud);

  // Motion Prior Cost
  struct MotionPriorCost {
    MotionPriorCost(const double* prior_pose, double weight);

    template <typename T>
    bool operator()(const T* const pose, T* residuals) const {
        residuals[0] = T(weight_) * (pose[0] - T(prior_pose_[0]));
        residuals[1] = T(weight_) * (pose[1] - T(prior_pose_[1]));
        residuals[2] = T(weight_) * (pose[2] - T(prior_pose_[2]));
        
        // Position difference
        residuals[3] = T(weight_) * (pose[3] - T(prior_pose_[3]));
        residuals[4] = T(weight_) * (pose[4] - T(prior_pose_[4]));
        residuals[5] = T(weight_) * (pose[5] - T(prior_pose_[5]));

        return true;
    }
  private:
    const double* prior_pose_;
    double weight_;
  };

  // Point to Point cost
  struct PointToPointCostFunction{
    PointToPointCostFunction(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr &source,
                             const pcl::PointCloud<pcl::PointXYZRGB>::Ptr &target,
                             double max_corr_dist);
      template <typename T>
      bool operator()(const T* const pose, T* residuals) const {

          // Check for NaN/inf in input pose
          for (int i = 0; i < 6; ++i) {
              if (!ceres::isfinite(pose[i])) {
                  return false;
              }
          }

          // Convert angle-axis to rotation matrix
          Eigen::Matrix<T, 3, 1> angle_axis(pose[0], pose[1], pose[2]);
          Eigen::Matrix<T, 3, 3> R;

          T angle = angle_axis.norm();
          const T kMinAngle = T(1e-8);  // Minimum rotation threshold
          
          if (angle > kMinAngle) {
              Eigen::Matrix<T, 3, 1> axis = angle_axis / angle;
              R = Eigen::AngleAxis<T>(angle, axis).toRotationMatrix();
          } else {
              R.setIdentity();
          }

          const T min_valid_residual = T(1e-6);

          for (size_t i = 0; i < correspondences_.size(); ++i) {
            const auto& src_pt = source_->points[i];
            const auto& tgt_pt = target_->points[correspondences_[i]];

            if (!pcl::isFinite(src_pt) || !pcl::isFinite(tgt_pt)) {
                residuals[i] = T(0);
                continue;
            }

            // Transform source point
            Eigen::Matrix<T, 3, 1> transformed_pt = 
                R * Eigen::Matrix<T, 3, 1>(T(src_pt.x), T(src_pt.y), T(src_pt.z)) +
                Eigen::Matrix<T, 3, 1>(pose[3], pose[4], pose[5]);

            // Calculate point-to-point distance
            Eigen::Matrix<T, 3, 1> point_diff = transformed_pt - 
                Eigen::Matrix<T, 3, 1>(T(tgt_pt.x), T(tgt_pt.y), T(tgt_pt.z));
            
            T distance_sq = point_diff.squaredNorm();
            
            // Apply threshold and compute residual
            if (distance_sq > T(max_corr_dist_sq_p2p_)) {
                residuals[i] = min_valid_residual;
            } else {
                residuals[i] = ceres::sqrt(distance_sq + min_valid_residual);
            }
          }
          return true;
      }
      size_t getNumCorrespondences() const { return correspondences_.size(); }
    private:
      void establishCorrespondencesP2P();
      pcl::PointCloud<pcl::PointXYZRGB>::Ptr source_;
      pcl::PointCloud<pcl::PointXYZRGB>::Ptr target_;
      std::vector<int> correspondences_;
      double max_corr_dist_sq_p2p_;
  };
  
  // GICP cost function for point clouds
  struct GICPCostFunction {
      GICPCostFunction(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& source_cloud,
                      const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& target_cloud,
                      const pcl::PointCloud<pcl::Normal>::Ptr& target_normals,
                      double max_corr_dist);
      template <typename T>
      bool operator()(const T* const pose, T* residuals) const {
          // 1. Convert angle-axis to rotation matrix (with safe normalization)
          
          // Check for NaN/inf in input pose
          for (int i = 0; i < 6; ++i) {
              if (!ceres::isfinite(pose[i])) {
                  return false;
              }
          }

          Eigen::Matrix<T, 3, 1> angle_axis(pose[0], pose[1], pose[2]);
          Eigen::Matrix<T, 3, 3> R;
          
          T angle = angle_axis.norm();
          const T kMinAngle = T(1e-8);  // Minimum rotation threshold
          
          if (angle > kMinAngle) {
              Eigen::Matrix<T, 3, 1> axis = angle_axis / angle;
              R = Eigen::AngleAxis<T>(angle, axis).toRotationMatrix();
          } else {
              R.setIdentity();
          }

          // 2. Residual scaling factors
          const T residual_scaling = T(1.0);  // Scale residuals to avoid tiny values
          const T max_valid_distance = T(max_corr_dist_sq_);  // Reject outliers
          const T min_valid_residual = T(1e-4);  // New: Minimum residual magnitude
          const T normal_length_threshold = T(0.1);
          const size_t n = correspondences_.size();
          
          for (size_t i = 0; i < correspondences_.size(); ++i) {
              // 3. Point validity checks
              const auto& src_pt = source_cloud_->points[i];
              const auto& tgt_pt = target_cloud_->points[correspondences_[i]];
              const auto& tgt_normal = target_normals_->points[correspondences_[i]];

              if (!pcl::isFinite(src_pt) || !pcl::isFinite(tgt_pt) || !pcl::isFinite(tgt_normal)) {
                  residuals[i] = T(0);
                  residuals[n + i] = T(0);
                  continue;
              }

              // 4. Transform source point
              Eigen::Matrix<T, 3, 1> transformed_pt = 
                  R * Eigen::Matrix<T, 3, 1>(T(src_pt.x), T(src_pt.y), T(src_pt.z)) +
                  Eigen::Matrix<T, 3, 1>(pose[3], pose[4], pose[5]);

              // 5. Calculate point-to-plane distance with outlier rejection
              Eigen::Matrix<T, 3, 1> point_diff = transformed_pt - 
                  Eigen::Matrix<T, 3, 1>(T(tgt_pt.x), T(tgt_pt.y), T(tgt_pt.z));
              
              T distance_sq = point_diff.squaredNorm();
              if (distance_sq > max_valid_distance) {
                  residuals[i] = min_valid_residual;
                  continue;
              }

              // 6. Scaled residual calculation with normal alignment check
              Eigen::Matrix<T, 3, 1> normal(T(tgt_normal.normal_x), 
                                          T(tgt_normal.normal_y), 
                                          T(tgt_normal.normal_z));

              T normal_length = normal.norm();
              if (normal_length < normal_length_threshold) {
                residuals[i] = min_valid_residual;
                continue;
              }

              normal = normal / normal_length;

              // Additional plane quality check using curvature
              const T curvature = T(tgt_normal.curvature);
              const T max_curvature = T(0.05); // Reject very curved surfaces

              if (curvature > max_curvature) {
                  residuals[i] = min_valid_residual;
                  continue;
              }

              // // Ensure normal is unit vector (numerical safety)
              // normal = normal.normalized();
              T residual_value  = point_diff.dot(normal);
              
              // 7. Apply soft thresholding
              residuals[i] = residual_scaling * ceres::sqrt(
                  residual_value * residual_value + min_valid_residual
              );

              // T normal_length = normal.norm();
              // if (normal_length < T(1e-6)) {
              //     residuals[i] = T(0);
              // } else {
              //     normal = normal / normal_length;
              //     T residual_value = point_diff.dot(normal);
              //     residuals[i] = residual_scaling * ceres::sqrt(
              //         residual_value * residual_value + min_valid_residual
              //     );
              // }

              // Point to point calculation 
              // if (distance_sq < max_valid_distance) {
              //   residuals[n + i] = ceres::sqrt(distance_sq);
              // } else {
              //   residuals[n + i] = min_valid_residual;
              // }

              // residuals[i] = residual_scaling * point_diff.dot(normal);
          }
          return true;
      }
      size_t getNumCorrespondences() const { return correspondences_.size(); }
  private:
      // void filterNaNPoints();
      void establishCorrespondences();
      pcl::PointCloud<pcl::PointXYZRGB>::Ptr source_cloud_;
      pcl::PointCloud<pcl::PointXYZRGB>::Ptr target_cloud_;
      pcl::PointCloud<pcl::Normal>::Ptr target_normals_;
      std::vector<int> correspondences_;
      double max_corr_dist_sq_;
  };
  // Optical flow cost function
  struct FlowCostFunction {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW ;
    FlowCostFunction(const std::vector<cv::Point2f> &curr_pts_filtered,
                     const std::vector<cv::Point2f> &prev_pts_filtered,
                     const std::vector<int> &frame_idxs,
                     const std::vector<int> &cloud_idxs,
                     const pcl::PointCloud<pcl::PointXYZRGB>::Ptr &source_cloud,
                     const pcl::PointCloud<pcl::PointXYZRGB>::Ptr &target_cloud,
                     const Eigen::Matrix3f  &R_cl,
                     const Eigen::Vector3f  &P_cl,
                     const Eigen::Matrix3f  &R_corr,
                     const Eigen::Vector3f  &P_corr, 
                     const Eigen::Matrix3f &K,
                     double max_correspondence_dist
                     ); 
    template <typename T>
    bool operator() (const T* const pose, T*residuals) const {

        // Check for NaN/inf in input pose
        for (int i = 0; i < 6; ++i) {
            if (!ceres::isfinite(pose[i])) {
                return false;
            }
        }
        // Initialize all residuals to zero first
       const size_t residuals_size = 2 * frame_idxs_.size();  // Each point has x,y residuals
        // std::fill(residuals, residuals + 2 * frame_idxs_.size(), T(0));
        last_valid_count_ = 0;

        Eigen::Map<const Eigen::Matrix<T, 3, 1>> angle_axis(pose);       
        Eigen::Map<const Eigen::Matrix<T, 3, 1>> P_wi(pose + 3);         
        Eigen::Matrix<T, 3, 3> R_wi;
        T angle = angle_axis.norm();
        const T kMinAngle = T(1e-8);  // Minimum rotation threshold
          
        if (angle > kMinAngle) {
            Eigen::Matrix<T, 3, 1> axis = angle_axis / angle;
            R_wi = Eigen::AngleAxis<T>(angle, axis).toRotationMatrix();
        } else {
            R_wi.setIdentity();
        }

        // Compute camera-to-world transformation using the current pose
        Eigen::Matrix<T, 3, 3> R_cw = R_cl_.cast<T>() * R_wi.transpose();
        Eigen::Matrix<T, 3, 1> P_cw = -R_cl_.cast<T>() * R_wi.transpose() * P_wi + P_cl_.cast<T>();

        // // KD-tree search variable
        // std::vector<int> indices(1);
        // std::vector<float> distances(1);
        // size_t valid_count = 0;

        // for (size_t idx = 0; idx < source_cloud_->size(); idx++) {
        //   residuals[2*idx] = T(0);
        //   residuals[2*idx+1] = T(0);
        //   if (!pcl::isFinite(source_cloud_->points[idx])) continue;          
          
        //   // Find nearest neighbor in target cloud
        //   std::lock_guard<std::mutex> lock(kdtree_mutex_);
        //   if (kdtree_.nearestKSearch(source_cloud_->points[idx], 1, indices, distances) == 0) {
        //     continue;       // No correspondence found
        //   }

        //   // Validate correspondence
        //   if (distances[0] > T(max_correspondence_dist_)) continue;

        //   if (distances[0] < 0.2 && indices[0] >= 0 && indices[0] < target_cloud_->size()) {
        //     int tgt_idx = indices[0];
        //     if (tgt_idx < 0 || tgt_idx >= target_cloud_->size()) continue;
        //     if (!pcl::isFinite(target_cloud_->points[tgt_idx])) continue;

        //     // Validate frame index
        //     int obs_idx;
        //     try {
        //           obs_idx = frame_idxs_.at(idx);
        //           if (obs_idx < 0 || obs_idx >= prev_pts_.size() || obs_idx >= curr_pts_.size()) {
        //               continue;
        //           }
        //     } catch (const std::out_of_range& e) {
        //           continue;
        //     }

        //     T dx_obs = T(curr_pts_[obs_idx].x - prev_pts_[obs_idx].x);
        //     T dy_obs = T(curr_pts_[obs_idx].y - prev_pts_[obs_idx].y);
        //     const auto& pt = target_cloud_->points[tgt_idx];
        //     Eigen::Matrix<T, 3, 1> tgt_pt(T(pt.x), T(pt.y), T(pt.z));

        //     // Project to image place
        //     Eigen::Matrix<T, 3, 1> pc      = R_cw * tgt_pt + P_cw;
        //     Eigen::Matrix<T, 3, 1> img_pts = K_.cast<T>() * pc;

        //     // Skip points behind camera
        //     if (img_pts.z() <= T(1e-6)) continue;

        //     img_pts /= img_pts.z();

        //     // Compute and store residuals
        //     residuals[2*idx]   = (img_pts.x() - T(prev_pts_[obs_idx].x)) - dx_obs;
        //     residuals[2*idx+1] = (img_pts.y() - T(prev_pts_[obs_idx].y)) - dy_obs;
        //     last_valid_count_++;
        //   }
        // }

        for (size_t idx = 0; idx < frame_idxs_.size(); idx++) {
          // Safety check for residuals buffer
          if (2*idx + 1 >= residuals_size) {
              return false;  // Would indicate serious configuration error
          }
          residuals[2*idx] = T(0);
          residuals[2*idx+1] = T(0);

          // Validate frame index
          int obs_idx;
          try {
                obs_idx = frame_idxs_.at(idx);
                if (obs_idx < 0 || obs_idx >= prev_pts_.size() || obs_idx >= curr_pts_.size()) {
                    continue;
                }
          } catch (const std::out_of_range& e) {
                continue;
          }

          T dx_obs = T(curr_pts_[obs_idx].x - prev_pts_[obs_idx].x);
          T dy_obs = T(curr_pts_[obs_idx].y - prev_pts_[obs_idx].y);
          
          //
          int cloud_idx_ = cloud_idxs_[idx];
          if (cloud_idx_ < 0 || cloud_idx_ >= source_cloud_->points.size()) {
            continue;  // Skip invalid indices
          }

          if (!pcl::isFinite(source_cloud_->points[cloud_idx_])) {
            return false;
          }
          const auto& point = source_cloud_->points[cloud_idx_];
          Eigen::Matrix<T, 3, 1> src_pt(T(point.x), T(point.y), T(point.z));

          // Transform to the new pose and project to image
          Eigen::Matrix<T, 3, 1> pc_world  = R_cw * src_pt + P_cw;
          Eigen::Matrix<T, 3, 1> pc_next   = R_corr_.cast<T>() * pc_world + P_corr_.cast<T>();
          Eigen::Matrix<T, 3, 1> img_pts   = K_.cast<T>() * pc_next;

          // Skip points behind camera
          if (img_pts.z() <= T(1e-6)) continue;
          img_pts /= img_pts.z();

          T residual_x = (img_pts.x() - T(prev_pts_[obs_idx].x)) - dx_obs;
          T residual_y = (img_pts.y() - T(prev_pts_[obs_idx].y)) - dy_obs;

          if (abs(residual_x) <= 3 && abs(residual_y) <= 3) {
            residuals[2*idx]   = residual_x;
            residuals[2*idx+1] = residual_y;
            // residuals[2*idx]   /= T(100.0);
            // residuals[2*idx+1] /= T(100.0);
            last_valid_count_++;
          }
          // residuals[2*idx]   = residual_x;
          // residuals[2*idx+1] = residual_y;
          // last_valid_count_++;
        }
        return last_valid_count_ > 0;
    }
    int getValidCount() const { return last_valid_count_; }
    static constexpr int kNumResiduals = ceres::DYNAMIC;

  private:
    std::vector<cv::Point2f> curr_pts_;
    std::vector<cv::Point2f> prev_pts_;
    std::vector<int> frame_idxs_;
    std::vector<int> cloud_idxs_;
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr source_cloud_;
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr target_cloud_;
    Eigen::Matrix3f R_cl_;
    Eigen::Vector3f P_cl_;
    Eigen::Matrix3f R_corr_;
    Eigen::Vector3f P_corr_;
    std::vector<size_t> valid_indices_;
    Eigen::Matrix3f K_;
    double max_correspondence_dist_;
    pcl::KdTreeFLANN<pcl::PointXYZRGB> kdtree_;
    mutable int last_valid_count_ = 0;
    mutable std::mutex kdtree_mutex_; 
  };

  // Compute the target cloud normals
  pcl::PointCloud<pcl::Normal>::Ptr computeNormals(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr &cloud);

  void optimization(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr &source_cloud,
                    const pcl::PointCloud<pcl::PointXYZRGB>::Ptr &target_cloud,
                    const pcl::PointCloud<pcl::PointXYZRGB>::Ptr &transformed_cloud,
                    const pcl::PointCloud<pcl::Normal>::Ptr &target_normals,
                    const std::vector<cv::Point2f> &prev_pts,
                    const std::vector<cv::Point2f> &curr_pts,
                    const std::vector<int> &frame_idxs,
                    const std::vector<int> &cloud_idxs,
                    const Eigen::Matrix4f &T_corr,
                    int total_dist);

  void findFlowCorrespondences(const std::vector<cv::Point2f> &curr_pts_filtered,
                     const std::vector<cv::Point2f> &prev_pts_filtered,
                     const std::vector<int> &frame_idxs,
                     const pcl::PointCloud<pcl::PointXYZRGB>::Ptr &source_cloud,
                     const pcl::PointCloud<pcl::PointXYZRGB>::Ptr &target_cloud,
                     std::vector<int> &corr,
                     const std::vector<int> &cloud_idxs
  );

  int optical_est;
  bool first_write;
  ///////////////////////////////

  // TF
  std::shared_ptr<tf2_ros::TransformBroadcaster> br;

  // ROS Msgs
  nav_msgs::msg::Odometry odom_ros;
  geometry_msgs::msg::PoseStamped pose_ros;
  nav_msgs::msg::Path path_ros;
  geometry_msgs::msg::PoseArray kf_pose_ros;

  // Flags
  std::atomic<bool> liom_initialized;
  std::atomic<bool> first_valid_scan;
  std::atomic<bool> first_imu_received;
  std::atomic<bool> imu_calibrated;
  std::atomic<bool> submap_hasChanged;
  std::atomic<bool> gicp_hasConverged;
  std::atomic<bool> deskew_status;
  std::atomic<int>  deskew_size;
  std::atomic<bool> rgb_colorized;

  // Threads
  std::thread publish_thread;
  std::thread publish_keyframe_thread;
  std::thread metrics_thread;
  std::thread debug_thread;
  std::thread publish_rgb_thread;

  // Trajectory
  std::vector<std::pair<Eigen::Vector3f, Eigen::Quaternionf>> trajectory;
  double length_traversed;

  // Keyframes
  std::vector<std::pair<std::pair<Eigen::Vector3f, Eigen::Quaternionf>,
                        pcl::PointCloud<PointType>::ConstPtr>> keyframes;
  std::vector<rclcpp::Time> keyframe_timestamps;
  std::vector<std::shared_ptr<const nano_gicp::CovarianceList>> keyframe_normals;
  std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>> keyframe_transformations;
  std::mutex keyframes_mutex;

  // Sensor Type
  liom::SensorType sensor;

  // Frames
  std::string odom_frame;
  std::string baselink_frame;
  std::string lidar_frame;
  std::string imu_frame;

  // Preprocessing
  pcl::CropBox<PointType> crop;
  pcl::VoxelGrid<PointType> voxel;

  // Point Clouds
  pcl::PointCloud<PointType>::ConstPtr original_scan;
  pcl::PointCloud<PointType>::ConstPtr deskewed_scan;
  pcl::PointCloud<PointType>::ConstPtr current_scan;



  // Keyframes
  pcl::PointCloud<PointType>::ConstPtr keyframe_cloud;
  int num_processed_keyframes;

  pcl::ConvexHull<PointType> convex_hull;
  pcl::ConcaveHull<PointType> concave_hull;
  std::vector<int> keyframe_convex;
  std::vector<int> keyframe_concave;

  // Submap
  pcl::PointCloud<PointType>::ConstPtr submap_cloud;
  std::shared_ptr<const nano_gicp::CovarianceList> submap_normals;
  std::shared_ptr<const nanoflann::KdTreeFLANN<PointType>> submap_kdtree;

  std::vector<int> submap_kf_idx_curr;
  std::vector<int> submap_kf_idx_prev;

  bool new_submap_is_ready;
  std::future<void> submap_future;
  std::condition_variable submap_build_cv;
  bool main_loop_running;
  std::mutex main_loop_running_mutex;

  // Timestamps
  rclcpp::Time scan_header_stamp;
  double scan_stamp;
  double prev_scan_stamp;
  double scan_dt;
  std::vector<double> comp_times;
  std::vector<double> imu_rates;
  std::vector<double> lidar_rates;

  double first_scan_stamp;
  double elapsed_time;

  // GICP
  nano_gicp::NanoGICP<PointType, PointType> gicp;
  nano_gicp::NanoGICP<PointType, PointType> gicp_temp;

  // Transformations
  Eigen::Matrix4f T, T_prior, T_corr;
  Eigen::Quaternionf q_final;

  Eigen::Vector3f origin;

  struct Extrinsics {
    struct SE3 {
      Eigen::Vector3f t;
      Eigen::Matrix3f R;
    };
    SE3 baselink2imu;
    SE3 baselink2lidar;
    Eigen::Matrix4f baselink2imu_T;
    Eigen::Matrix4f baselink2lidar_T;
  }; Extrinsics extrinsics;


  struct Camera {
    int width;
    int height;
    float fx, fy, cx, cy;
    Eigen::Matrix3f intrinsic_matrix;
  }; Camera camera;

  // IMU
  rclcpp::Time imu_stamp;
  double first_imu_stamp;
  double prev_imu_stamp;
  double imu_dp, imu_dq_deg;

  struct ImuMeas {
    double stamp;
    double dt; // defined as the difference between the current and the previous measurement
    Eigen::Vector3f ang_vel;
    Eigen::Vector3f lin_accel;
  }; ImuMeas imu_meas;

  boost::circular_buffer<ImuMeas> imu_buffer;
  std::mutex mtx_imu;
  std::condition_variable cv_imu_stamp;

  static bool comparatorImu(ImuMeas m1, ImuMeas m2) {
    return (m1.stamp < m2.stamp);
  };

  // Geometric Observer
  struct Geo {
    bool first_opt_done;
    std::mutex mtx;
    double dp;
    double dq_deg;
    Eigen::Vector3f prev_p;
    Eigen::Quaternionf prev_q;
    Eigen::Vector3f prev_vel;
  }; Geo geo;

  // State Vector
  struct ImuBias {
    Eigen::Vector3f gyro;
    Eigen::Vector3f accel;
  };

  struct Frames {
    Eigen::Vector3f b;
    Eigen::Vector3f w;
  };

  struct Velocity {
    Frames lin;
    Frames ang;
  };

  struct State {
    Eigen::Vector3f p;    // position in world frame
    Eigen::Quaternionf q; // orientation in world frame
    Eigen::Matrix3f rot;  // rotation matrix in world frame
    Velocity v;
    ImuBias b;            // imu biases in body frame
  }; State state;

  struct Pose {
    Eigen::Vector3f p; // position in world frame
    Eigen::Quaternionf q; // orientation in world frame
  };
  Pose lidarPose;
  Pose imuPose;

  // Metrics
  struct Metrics {
    std::vector<float> spaciousness;
    std::vector<float> density;
  }; Metrics metrics;

  std::string cpu_type;
  std::vector<double> cpu_percents;
  clock_t lastCPU, lastSysCPU, lastUserCPU;
  int numProcessors;

  // Parameters
  std::string version_;
  int num_threads_;

  bool deskew_;

  double gravity_;

  bool time_offset_;

  bool adaptive_params_;

  double obs_submap_thresh_;
  double obs_keyframe_thresh_;
  double obs_keyframe_lag_;

  double keyframe_thresh_dist_;
  double keyframe_thresh_rot_;

  int submap_knn_;
  int submap_kcv_;
  int submap_kcc_;
  double submap_concave_alpha_;

  bool densemap_filtered_;
  bool wait_until_move_;

  double crop_size_;

  bool vf_use_;
  double vf_res_;

  bool imu_calibrate_;
  bool calibrate_gyro_;
  bool calibrate_accel_;
  bool gravity_align_;
  double imu_calib_time_;
  int imu_buffer_size_;
  Eigen::Matrix3f imu_accel_sm_;

  int gicp_min_num_points_;
  int gicp_k_correspondences_;
  double gicp_max_corr_dist_;
  int gicp_max_iter_;
  double gicp_transformation_ep_;
  double gicp_rotation_ep_;
  double gicp_init_lambda_factor_;

  double geo_Kp_;
  double geo_Kv_;
  double geo_Kq_;
  double geo_Kab_;
  double geo_Kgb_;
  double geo_abias_max_;
  double geo_gbias_max_;

};
