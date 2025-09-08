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

#include "liom/odom.h"
#include "liom/utils.h"

#include <queue>

#include "rclcpp/qos.hpp"
#include "rclcpp/rclcpp.hpp"

liom::OdomNode::OdomNode() : Node("liom_odom_node") {

  this->getParams();

  this->num_threads_ = omp_get_max_threads();

  this->liom_initialized = false;
  this->first_valid_scan = false;
  this->first_imu_received = false;
  if (this->imu_calibrate_) {this->imu_calibrated = false;}
  else {this->imu_calibrated = true;}
  this->deskew_status = false;
  this->deskew_size = 0;

  // Modification
  this->rgb_colorized = false;
  this->visual_initialization = false;
  this->valid_point_flag      = false;
  this->mahalanobis_idx_      = false;

  // Callback and Subscriber
  this->lidar_cb_group = this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
  auto lidar_sub_opt = rclcpp::SubscriptionOptions();
  lidar_sub_opt.callback_group = this->lidar_cb_group;
  this->lidar_sub = this->create_subscription<sensor_msgs::msg::PointCloud2>("pointcloud", 1,
      std::bind(&liom::OdomNode::callbackPointCloud, this, std::placeholders::_1), lidar_sub_opt);

  this->imu_cb_group = this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
  auto imu_sub_opt = rclcpp::SubscriptionOptions();
  imu_sub_opt.callback_group = this->imu_cb_group;
  this->imu_sub = this->create_subscription<sensor_msgs::msg::Imu>("imu", rclcpp::SensorDataQoS(),
      std::bind(&liom::OdomNode::callbackImu, this, std::placeholders::_1), imu_sub_opt);
  
  // Modification
  this->img_cb_group = this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
  auto img_sub_opt   = rclcpp::SubscriptionOptions();
  img_sub_opt.callback_group = this->img_cb_group;
  this->img_sub      = this->create_subscription<sensor_msgs::msg::Image>("image", rclcpp::SensorDataQoS(),
      std::bind(&liom::OdomNode::callbackImage, this, std::placeholders::_1), img_sub_opt);
  
  // Publisher
  this->odom_pub     = this->create_publisher<nav_msgs::msg::Odometry>("odom", 1);
  this->pose_pub     = this->create_publisher<geometry_msgs::msg::PoseStamped>("pose", 1);
  this->path_pub     = this->create_publisher<nav_msgs::msg::Path>("path", 1);
  this->kf_pose_pub  = this->create_publisher<geometry_msgs::msg::PoseArray>("kf_pose", 1);
  this->kf_cloud_pub = this->create_publisher<sensor_msgs::msg::PointCloud2>("kf_cloud", 1);
  this->deskewed_pub = this->create_publisher<sensor_msgs::msg::PointCloud2>("deskewed", 1);

  this->br = std::make_shared<tf2_ros::TransformBroadcaster>(*this);

  this->publish_timer = this->create_wall_timer(std::chrono::duration<double>(0.01), 
      std::bind(&liom::OdomNode::publishPose, this));
  
  // Additional Publisher
  this->pubImage = image_transport::create_publisher(this, "rgb_img");

  this->T = Eigen::Matrix4f::Identity();
  this->T_prior = Eigen::Matrix4f::Identity();
  this->T_corr = Eigen::Matrix4f::Identity();

  this->origin = Eigen::Vector3f(0., 0., 0.);
  this->state.p = Eigen::Vector3f(0., 0., 0.);
  this->state.q = Eigen::Quaternionf(1., 0., 0., 0.);
  this->state.v.lin.b = Eigen::Vector3f(0., 0., 0.);
  this->state.v.lin.w = Eigen::Vector3f(0., 0., 0.);
  this->state.v.ang.b = Eigen::Vector3f(0., 0., 0.);
  this->state.v.ang.w = Eigen::Vector3f(0., 0., 0.);

  this->lidarPose.p = Eigen::Vector3f(0., 0., 0.);
  this->lidarPose.q = Eigen::Quaternionf(1., 0., 0., 0.);

  this->imu_meas.stamp = 0.;
  this->imu_meas.ang_vel[0] = 0.;
  this->imu_meas.ang_vel[1] = 0.;
  this->imu_meas.ang_vel[2] = 0.;
  this->imu_meas.lin_accel[0] = 0.;
  this->imu_meas.lin_accel[1] = 0.;
  this->imu_meas.lin_accel[2] = 0.;

  this->imu_buffer.set_capacity(this->imu_buffer_size_);
  this->first_imu_stamp = 0.;
  this->prev_imu_stamp = 0.;

  this->original_scan = std::make_shared<const pcl::PointCloud<PointType>>();
  this->deskewed_scan = std::make_shared<const pcl::PointCloud<PointType>>();
  this->current_scan = std::make_shared<const pcl::PointCloud<PointType>>();
  this->submap_cloud = std::make_shared<const pcl::PointCloud<PointType>>();

  this->num_processed_keyframes = 0;

  this->submap_hasChanged = true;
  this->submap_kf_idx_prev.clear();

  this->first_scan_stamp = 0.;
  this->elapsed_time = 0.;
  this->length_traversed;

  this->convex_hull.setDimension(3);
  this->concave_hull.setDimension(3);
  this->concave_hull.setAlpha(this->keyframe_thresh_dist_);
  this->concave_hull.setKeepInformation(true);

  this->gicp.setCorrespondenceRandomness(this->gicp_k_correspondences_);
  this->gicp.setMaxCorrespondenceDistance(this->gicp_max_corr_dist_);
  this->gicp.setMaximumIterations(this->gicp_max_iter_);
  this->gicp.setTransformationEpsilon(this->gicp_transformation_ep_);
  this->gicp.setRotationEpsilon(this->gicp_rotation_ep_);
  this->gicp.setInitialLambdaFactor(this->gicp_init_lambda_factor_);

  this->gicp_temp.setCorrespondenceRandomness(this->gicp_k_correspondences_);
  this->gicp_temp.setMaxCorrespondenceDistance(this->gicp_max_corr_dist_);
  this->gicp_temp.setMaximumIterations(this->gicp_max_iter_);
  this->gicp_temp.setTransformationEpsilon(this->gicp_transformation_ep_);
  this->gicp_temp.setRotationEpsilon(this->gicp_rotation_ep_);
  this->gicp_temp.setInitialLambdaFactor(this->gicp_init_lambda_factor_);

  pcl::Registration<PointType, PointType>::KdTreeReciprocalPtr temp;
  this->gicp.setSearchMethodSource(temp, true);
  this->gicp.setSearchMethodTarget(temp, true);
  this->gicp_temp.setSearchMethodSource(temp, true);
  this->gicp_temp.setSearchMethodTarget(temp, true);

  this->geo.first_opt_done = false;
  this->geo.prev_vel = Eigen::Vector3f(0., 0., 0.);

  pcl::console::setVerbosityLevel(pcl::console::L_ERROR);

  this->crop.setNegative(true);
  this->crop.setMin(Eigen::Vector4f(-this->crop_size_, -this->crop_size_, -this->crop_size_, 1.0));
  this->crop.setMax(Eigen::Vector4f(this->crop_size_, this->crop_size_, this->crop_size_, 1.0));

  this->voxel.setLeafSize(this->vf_res_, this->vf_res_, this->vf_res_);

  this->metrics.spaciousness.push_back(0.);
  this->metrics.density.push_back(this->gicp_max_corr_dist_);

  // Modification
  this->last_timestamp_img = 0.0;
  this->lid_num            = 0.0;
  this->first_write        = true;

  /*
    M2DGR dataset
  */
  this->camera.width = 640; this->camera.height = 480;
  this->camera.fx = 617.971050917033;
  this->camera.fy = 616.445131524790;
  this->camera.cx = 327.710279392468;
  this->camera.cy = 253.976983707814;

  /*
    Cam-head to LiDAR
  */ 
  this->Rcl << 0, -1, 0,
               0, 0, -1, 
               1, 0, 0;

  this->Pcl << 0.00065,  0.65376, -0.30456; 

  // Camera instrinsic params
  this->camera.intrinsic_matrix << this->camera.fx, 0, this->camera.cx,
                                    0, this->camera.fy, this->camera.cy,
                                    0, 0, 1;


  // CPU Specs
  char CPUBrandString[0x40];
  memset(CPUBrandString, 0, sizeof(CPUBrandString));

  this->cpu_type = "";

  #ifdef HAS_CPUID
  unsigned int CPUInfo[4] = {0,0,0,0};
  __cpuid(0x80000000, CPUInfo[0], CPUInfo[1], CPUInfo[2], CPUInfo[3]);
  unsigned int nExIds = CPUInfo[0];
  for (unsigned int i = 0x80000000; i <= nExIds; ++i) {
    __cpuid(i, CPUInfo[0], CPUInfo[1], CPUInfo[2], CPUInfo[3]);
    if (i == 0x80000002)
      memcpy(CPUBrandString, CPUInfo, sizeof(CPUInfo));
    else if (i == 0x80000003)
      memcpy(CPUBrandString + 16, CPUInfo, sizeof(CPUInfo));
    else if (i == 0x80000004)
      memcpy(CPUBrandString + 32, CPUInfo, sizeof(CPUInfo));
  }
  this->cpu_type = CPUBrandString;
  boost::trim(this->cpu_type);
  #endif

  FILE* file;
  struct tms timeSample;
  char line[128];

  this->lastCPU = times(&timeSample);
  this->lastSysCPU = timeSample.tms_stime;
  this->lastUserCPU = timeSample.tms_utime;

  file = fopen("/proc/cpuinfo", "r");
  this->numProcessors = 0;
  while(fgets(line, 128, file) != nullptr) {
      if (strncmp(line, "processor", 9) == 0) this->numProcessors++;
  }
  fclose(file);

}

liom::OdomNode::~OdomNode() {}

void liom::OdomNode::getParams() {

  // Version
  liom::declare_param(this, "version", this->version_, "0.0.0");

  // Frames
  liom::declare_param(this, "frames/odom", this->odom_frame, "odom");
  liom::declare_param(this, "frames/baselink", this->baselink_frame, "base_link");
  liom::declare_param(this, "frames/lidar", this->lidar_frame, "lidar");
  liom::declare_param(this, "frames/imu", this->imu_frame, "imu");

  // Deskew Flag
  liom::declare_param(this, "pointcloud/deskew", this->deskew_, true);

  // Gravity
  liom::declare_param(this, "odom/gravity", this->gravity_, 9.80665);

  // Compute time offset between lidar and imu
  liom::declare_param(this, "odom/computeTimeOffset", this->time_offset_, false);

  // Keyframe Threshold
  liom::declare_param(this, "odom/keyframe/threshD", this->keyframe_thresh_dist_, 0.1);
  liom::declare_param(this, "odom/keyframe/threshR", this->keyframe_thresh_rot_, 1.0);

  // Submap
  liom::declare_param(this, "odom/submap/keyframe/knn", this->submap_knn_, 10);
  liom::declare_param(this, "odom/submap/keyframe/kcv", this->submap_kcv_, 10);
  liom::declare_param(this, "odom/submap/keyframe/kcc", this->submap_kcc_, 10);

  // Dense map resolution
  liom::declare_param(this, "map/dense/filtered", this->densemap_filtered_, true);

  // Wait until movement to publish map
  liom::declare_param(this, "map/waitUntilMove", this->wait_until_move_, false);

  // Crop Box Filter
  liom::declare_param(this, "odom/preprocessing/cropBoxFilter/size", this->crop_size_, 1.0);

  // Voxel Grid Filter
  liom::declare_param(this, "pointcloud/voxelize", this->vf_use_, true);
  liom::declare_param(this, "odom/preprocessing/voxelFilter/res", this->vf_res_, 0.05);

  // Adaptive Parameters
  liom::declare_param(this, "adaptive", this->adaptive_params_, true);

  // Extrinsics
  std::vector<double> t_default{0., 0., 0.};
  std::vector<double> R_default{1., 0., 0., 0., 1., 0., 0., 0., 1.};

  // center of gravity to imu
  std::vector<double> baselink2imu_t, baselink2imu_R;
  liom::declare_param(this, "extrinsics/baselink2imu/t", baselink2imu_t, t_default);
  liom::declare_param(this, "extrinsics/baselink2imu/R", baselink2imu_R, R_default);
  this->extrinsics.baselink2imu.t =
    Eigen::Vector3f(baselink2imu_t[0], baselink2imu_t[1], baselink2imu_t[2]);
  this->extrinsics.baselink2imu.R =
    Eigen::Map<const Eigen::Matrix<float, -1, -1, Eigen::RowMajor>>(std::vector<float>(baselink2imu_R.begin(), baselink2imu_R.end()).data(), 3, 3);
  this->extrinsics.baselink2imu_T = Eigen::Matrix4f::Identity();
  this->extrinsics.baselink2imu_T.block(0, 3, 3, 1) = this->extrinsics.baselink2imu.t;
  this->extrinsics.baselink2imu_T.block(0, 0, 3, 3) = this->extrinsics.baselink2imu.R;

  // center of gravity to lidar
  std::vector<double> baselink2lidar_t, baselink2lidar_R;
  liom::declare_param(this, "extrinsics/baselink2lidar/t", baselink2lidar_t, t_default);
  liom::declare_param(this, "extrinsics/baselink2lidar/R", baselink2lidar_R, R_default);

  this->extrinsics.baselink2lidar.t =
    Eigen::Vector3f(baselink2lidar_t[0], baselink2lidar_t[1], baselink2lidar_t[2]);
  this->extrinsics.baselink2lidar.R =
    Eigen::Map<const Eigen::Matrix<float, -1, -1, Eigen::RowMajor>>(std::vector<float>(baselink2lidar_R.begin(), baselink2lidar_R.end()).data(), 3, 3);

  this->extrinsics.baselink2lidar_T = Eigen::Matrix4f::Identity();
  this->extrinsics.baselink2lidar_T.block(0, 3, 3, 1) = this->extrinsics.baselink2lidar.t;
  this->extrinsics.baselink2lidar_T.block(0, 0, 3, 3) = this->extrinsics.baselink2lidar.R;

  // IMU
  liom::declare_param(this, "odom/imu/calibration/accel", this->calibrate_accel_, true);
  liom::declare_param(this, "odom/imu/calibration/gyro", this->calibrate_gyro_, true);
  liom::declare_param(this, "odom/imu/calibration/time", this->imu_calib_time_, 3.0);
  liom::declare_param(this, "odom/imu/bufferSize", this->imu_buffer_size_, 2000);

  std::vector<double> accel_default{0., 0., 0.}; std::vector<double> prior_accel_bias;
  std::vector<double> gyro_default{0., 0., 0.}; std::vector<double> prior_gyro_bias;

  liom::declare_param(this, "odom/imu/approximateGravity", this->gravity_align_, true);
  liom::declare_param(this, "imu/calibration", this->imu_calibrate_, true);
  liom::declare_param(this, "imu/intrinsics/accel/bias", prior_accel_bias, accel_default);
  liom::declare_param(this, "imu/intrinsics/gyro/bias", prior_gyro_bias, gyro_default);

  // scale-misalignment matrix
  std::vector<double> imu_sm_default{1., 0., 0., 0., 1., 0., 0., 0., 1.};
  std::vector<double> imu_sm;

  liom::declare_param(this, "imu/intrinsics/accel/sm", imu_sm, imu_sm_default);

  if (!this->imu_calibrate_) {
    this->state.b.accel[0] = prior_accel_bias[0];
    this->state.b.accel[1] = prior_accel_bias[1];
    this->state.b.accel[2] = prior_accel_bias[2];
    this->state.b.gyro[0] = prior_gyro_bias[0];
    this->state.b.gyro[1] = prior_gyro_bias[1];
    this->state.b.gyro[2] = prior_gyro_bias[2];
    this->imu_accel_sm_ = Eigen::Map<const Eigen::Matrix<float, -1, -1, Eigen::RowMajor>>(std::vector<float>(imu_sm.begin(), imu_sm.end()).data(), 3, 3);
  } else {
    this->state.b.accel = Eigen::Vector3f(0., 0., 0.);
    this->state.b.gyro = Eigen::Vector3f(0., 0., 0.);
    this->imu_accel_sm_ = Eigen::Matrix3f::Identity();
  }

  // GICP
  liom::declare_param(this, "odom/gicp/minNumPoints", this->gicp_min_num_points_, 100);
  liom::declare_param(this, "odom/gicp/kCorrespondences", this->gicp_k_correspondences_, 20);
  liom::declare_param(this, "odom/gicp/maxCorrespondenceDistance", this->gicp_max_corr_dist_,
      std::sqrt(std::numeric_limits<double>::max()));
  liom::declare_param(this, "odom/gicp/maxIterations", this->gicp_max_iter_, 64);
  liom::declare_param(this, "odom/gicp/transformationEpsilon", this->gicp_transformation_ep_, 0.0005);
  liom::declare_param(this, "odom/gicp/rotationEpsilon", this->gicp_rotation_ep_, 0.0005);
  liom::declare_param(this, "odom/gicp/initLambdaFactor", this->gicp_init_lambda_factor_, 1e-9);

  // Geometric Observer
  liom::declare_param(this, "odom/geo/Kp", this->geo_Kp_, 1.0);
  liom::declare_param(this, "odom/geo/Kv", this->geo_Kv_, 1.0);
  liom::declare_param(this, "odom/geo/Kq", this->geo_Kq_, 1.0);
  liom::declare_param(this, "odom/geo/Kab", this->geo_Kab_, 1.0);
  liom::declare_param(this, "odom/geo/Kgb", this->geo_Kgb_, 1.0);
  liom::declare_param(this, "odom/geo/abias_max", this->geo_abias_max_, 1.0);
  liom::declare_param(this, "odom/geo/gbias_max", this->geo_gbias_max_, 1.0);
}

void liom::OdomNode::start() {

  printf("\033[2J\033[1;1H");
  std::cout << std::endl
            << "+-------------------------------------------------------------------+" << std::endl;
  std::cout << "+-------------------------------------------------------------------+" << std::endl;

}

void liom::OdomNode::publishPose() {

  // nav_msgs::msg::Odometry
  this->odom_ros.header.stamp = this->imu_stamp;
  this->odom_ros.header.frame_id = this->odom_frame;
  this->odom_ros.child_frame_id = this->baselink_frame;

  this->odom_ros.pose.pose.position.x = this->state.p[0];
  this->odom_ros.pose.pose.position.y = this->state.p[1];
  this->odom_ros.pose.pose.position.z = this->state.p[2];

  this->odom_ros.pose.pose.orientation.w = this->state.q.w();
  this->odom_ros.pose.pose.orientation.x = this->state.q.x();
  this->odom_ros.pose.pose.orientation.y = this->state.q.y();
  this->odom_ros.pose.pose.orientation.z = this->state.q.z();

  this->odom_ros.twist.twist.linear.x = this->state.v.lin.w[0];
  this->odom_ros.twist.twist.linear.y = this->state.v.lin.w[1];
  this->odom_ros.twist.twist.linear.z = this->state.v.lin.w[2];

  this->odom_ros.twist.twist.angular.x = this->state.v.ang.b[0];
  this->odom_ros.twist.twist.angular.y = this->state.v.ang.b[1];
  this->odom_ros.twist.twist.angular.z = this->state.v.ang.b[2];

  this->odom_pub->publish(this->odom_ros);

  // geometry_msgs::msg::PoseStamped
  this->pose_ros.header.stamp = this->imu_stamp;
  this->pose_ros.header.frame_id = this->odom_frame;

  this->pose_ros.pose.position.x = this->state.p[0];
  this->pose_ros.pose.position.y = this->state.p[1];
  this->pose_ros.pose.position.z = this->state.p[2];

  this->pose_ros.pose.orientation.w = this->state.q.w();
  this->pose_ros.pose.orientation.x = this->state.q.x();
  this->pose_ros.pose.orientation.y = this->state.q.y();
  this->pose_ros.pose.orientation.z = this->state.q.z();

  this->pose_pub->publish(this->pose_ros);

}

void liom::OdomNode::publishToROS(pcl::PointCloud<PointType>::ConstPtr published_cloud, Eigen::Matrix4f T_cloud) {
  this->publishCloud(published_cloud, T_cloud);

  // nav_msgs::msg::Path
  this->path_ros.header.stamp = this->imu_stamp;
  this->path_ros.header.frame_id = this->odom_frame;

  geometry_msgs::msg::PoseStamped p;
  p.header.stamp = this->imu_stamp;
  p.header.frame_id = this->odom_frame;
  p.pose.position.x = this->state.p[0];
  p.pose.position.y = this->state.p[1];
  p.pose.position.z = this->state.p[2];
  p.pose.orientation.w = this->state.q.w();
  p.pose.orientation.x = this->state.q.x();
  p.pose.orientation.y = this->state.q.y();
  p.pose.orientation.z = this->state.q.z();

  this->path_ros.poses.push_back(p);
  this->path_pub->publish(this->path_ros);

  // transform: odom to baselink
  geometry_msgs::msg::TransformStamped transformStamped;

  transformStamped.header.stamp = this->imu_stamp;
  transformStamped.header.frame_id = this->odom_frame;
  transformStamped.child_frame_id = this->baselink_frame;

  transformStamped.transform.translation.x = this->state.p[0];
  transformStamped.transform.translation.y = this->state.p[1];
  transformStamped.transform.translation.z = this->state.p[2];

  transformStamped.transform.rotation.w = this->state.q.w();
  transformStamped.transform.rotation.x = this->state.q.x();
  transformStamped.transform.rotation.y = this->state.q.y();
  transformStamped.transform.rotation.z = this->state.q.z();

  br->sendTransform(transformStamped);

  // transform: baselink to imu
  transformStamped.header.stamp = this->imu_stamp;
  transformStamped.header.frame_id = this->baselink_frame;
  transformStamped.child_frame_id = this->imu_frame;

  transformStamped.transform.translation.x = this->extrinsics.baselink2imu.t[0];
  transformStamped.transform.translation.y = this->extrinsics.baselink2imu.t[1];
  transformStamped.transform.translation.z = this->extrinsics.baselink2imu.t[2];

  Eigen::Quaternionf q(this->extrinsics.baselink2imu.R);
  transformStamped.transform.rotation.w = q.w();
  transformStamped.transform.rotation.x = q.x();
  transformStamped.transform.rotation.y = q.y();
  transformStamped.transform.rotation.z = q.z();

  br->sendTransform(transformStamped);

  // transform: baselink to lidar
  transformStamped.header.stamp = this->imu_stamp;
  transformStamped.header.frame_id = this->baselink_frame;
  transformStamped.child_frame_id = this->lidar_frame;

  transformStamped.transform.translation.x = this->extrinsics.baselink2lidar.t[0];
  transformStamped.transform.translation.y = this->extrinsics.baselink2lidar.t[1];
  transformStamped.transform.translation.z = this->extrinsics.baselink2lidar.t[2];

  Eigen::Quaternionf qq(this->extrinsics.baselink2lidar.R);
  transformStamped.transform.rotation.w = qq.w();
  transformStamped.transform.rotation.x = qq.x();
  transformStamped.transform.rotation.y = qq.y();
  transformStamped.transform.rotation.z = qq.z();

  br->sendTransform(transformStamped);

}

void liom::OdomNode::publishCloud(pcl::PointCloud<PointType>::ConstPtr published_cloud, Eigen::Matrix4f T_cloud) {

  if (this->wait_until_move_) {
    if (this->length_traversed < 0.1) { return; }
  }

  pcl::PointCloud<PointType>::Ptr deskewed_scan_t_ = std::make_shared<pcl::PointCloud<PointType>>();

  pcl::transformPointCloud (*published_cloud, *deskewed_scan_t_, T_cloud);

  // published deskewed cloud
  sensor_msgs::msg::PointCloud2 deskewed_ros;
  pcl::toROSMsg(*deskewed_scan_t_, deskewed_ros);
  deskewed_ros.header.stamp = this->scan_header_stamp;
  deskewed_ros.header.frame_id = this->odom_frame;
  this->deskewed_pub->publish(deskewed_ros);

}

void liom::OdomNode::publishKeyframe(std::pair<std::pair<Eigen::Vector3f, Eigen::Quaternionf>, pcl::PointCloud<PointType>::ConstPtr> kf, rclcpp::Time timestamp) {

  // Push back
  geometry_msgs::msg::Pose p;
  p.position.x = kf.first.first[0];
  p.position.y = kf.first.first[1];
  p.position.z = kf.first.first[2];
  p.orientation.w = kf.first.second.w();
  p.orientation.x = kf.first.second.x();
  p.orientation.y = kf.first.second.y();
  p.orientation.z = kf.first.second.z();
  this->kf_pose_ros.poses.push_back(p);

  // Publish
  this->kf_pose_ros.header.stamp = timestamp;
  this->kf_pose_ros.header.frame_id = this->odom_frame;
  this->kf_pose_pub->publish(this->kf_pose_ros);

  // publish keyframe scan for map
  if (this->vf_use_) {
    if (kf.second->points.size() == kf.second->width * kf.second->height) {
      sensor_msgs::msg::PointCloud2 keyframe_cloud_ros;
      pcl::toROSMsg(*kf.second, keyframe_cloud_ros);
      keyframe_cloud_ros.header.stamp = timestamp;
      keyframe_cloud_ros.header.frame_id = this->odom_frame;
      this->kf_cloud_pub->publish(keyframe_cloud_ros);
    }
  } else {
    sensor_msgs::msg::PointCloud2 keyframe_cloud_ros;
    pcl::toROSMsg(*kf.second, keyframe_cloud_ros);
    keyframe_cloud_ros.header.stamp = timestamp;
    keyframe_cloud_ros.header.frame_id = this->odom_frame;
    this->kf_cloud_pub->publish(keyframe_cloud_ros);
  }

}

void liom::OdomNode::getScanFromROS(const sensor_msgs::msg::PointCloud2::SharedPtr& pc) {

  pcl::PointCloud<PointType>::Ptr original_scan_ = std::make_shared<pcl::PointCloud<PointType>>();
  pcl::fromROSMsg(*pc, *original_scan_);

  // Remove NaNs
  std::vector<int> idx;
  original_scan_->is_dense = false;
  pcl::removeNaNFromPointCloud(*original_scan_, *original_scan_, idx);

  // Crop Box Filter
  this->crop.setInputCloud(original_scan_);
  this->crop.filter(*original_scan_);

  // automatically detect sensor type
  this->sensor = liom::SensorType::UNKNOWN;
  for (auto &field : pc->fields) {
    if (field.name == "t") {
      this->sensor = liom::SensorType::OUSTER;
      break;
    } else if (field.name == "time") {
      this->sensor = liom::SensorType::VELODYNE;
      break;
    } else if (field.name == "timestamp" && original_scan_->points[0].timestamp < 1e14) {
      this->sensor = liom::SensorType::HESAI;
      break;
    } else if (field.name == "timestamp" && original_scan_->points[0].timestamp > 1e14) {
      this->sensor = liom::SensorType::LIVOX;
      break;
    }
  }

  if (this->sensor == liom::SensorType::UNKNOWN) {
    this->deskew_ = false;
  }

  this->scan_header_stamp = pc->header.stamp;
  this->original_scan = original_scan_;

}

void liom::OdomNode::preprocessPoints() {

  // Deskew the original liom-type scan
  if (this->deskew_) {

    this->deskewPointcloud();

    if (!this->first_valid_scan) {
      return;
    }

  } else {

    this->scan_stamp = rclcpp::Time(this->scan_header_stamp).seconds();

    // don't process scans until IMU data is present
    if (!this->first_valid_scan) {

      if (this->imu_buffer.empty() || this->scan_stamp <= this->imu_buffer.back().stamp) {
        return;
      }

      this->first_valid_scan = true;
      this->T_prior = this->T; // assume no motion for the first scan

    } else {

      // IMU prior for second scan onwards
    std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>> frames;
    frames = this->integrateImu(this->prev_scan_stamp, this->lidarPose.q, this->lidarPose.p,
                                this->geo.prev_vel.cast<float>(), {this->scan_stamp});

    if (frames.size() > 0) {
      this->T_prior = frames.back();
    } else {
      this->T_prior = this->T;
    }

    }

    pcl::PointCloud<PointType>::Ptr deskewed_scan_ = std::make_shared<pcl::PointCloud<PointType>>();
    pcl::transformPointCloud (*this->original_scan, *deskewed_scan_,
                              this->T_prior * this->extrinsics.baselink2lidar_T);
    this->deskewed_scan = deskewed_scan_;
    this->deskew_status = false;
  }

  // Voxel Grid Filter
  if (this->vf_use_) {
    pcl::PointCloud<PointType>::Ptr current_scan_ = std::make_shared<pcl::PointCloud<PointType>>(*this->deskewed_scan);
    this->voxel.setInputCloud(current_scan_);
    this->voxel.filter(*current_scan_);
    this->current_scan = current_scan_;
  } else {
    this->current_scan = this->deskewed_scan;
  }

}

void liom::OdomNode::deskewPointcloud() {

  pcl::PointCloud<PointType>::Ptr deskewed_scan_ = std::make_shared<pcl::PointCloud<PointType>>(1, this->original_scan->points.size());
  // individual point timestamps should be relative to this time
  double sweep_ref_time = rclcpp::Time(this->scan_header_stamp).seconds();

  // sort points by timestamp and build list of timestamps
  std::function<bool(const PointType&, const PointType&)> point_time_cmp;
  std::function<bool(boost::range::index_value<PointType&, long>,
                     boost::range::index_value<PointType&, long>)> point_time_neq;
  std::function<double(boost::range::index_value<PointType&, long>)> extract_point_time;

  if (this->sensor == liom::SensorType::OUSTER) {

    point_time_cmp = [](const PointType& p1, const PointType& p2)
      { return p1.t < p2.t; };
    point_time_neq = [](boost::range::index_value<PointType&, long> p1,
                        boost::range::index_value<PointType&, long> p2)
      { return p1.value().t != p2.value().t; };
    extract_point_time = [&sweep_ref_time](boost::range::index_value<PointType&, long> pt)
      { return sweep_ref_time + pt.value().t * 1e-9f; };

  } else if (this->sensor == liom::SensorType::VELODYNE) {

    point_time_cmp = [](const PointType& p1, const PointType& p2)
      { return p1.time < p2.time; };
    point_time_neq = [](boost::range::index_value<PointType&, long> p1,
                        boost::range::index_value<PointType&, long> p2)
      { return p1.value().time != p2.value().time; };
    extract_point_time = [&sweep_ref_time](boost::range::index_value<PointType&, long> pt)
      { return sweep_ref_time + pt.value().time; };

  } else if (this->sensor == liom::SensorType::HESAI) {

    point_time_cmp = [](const PointType& p1, const PointType& p2)
      { return p1.timestamp < p2.timestamp; };
    point_time_neq = [](boost::range::index_value<PointType&, long> p1,
                        boost::range::index_value<PointType&, long> p2)
      { return p1.value().timestamp != p2.value().timestamp; };
    extract_point_time = [&sweep_ref_time](boost::range::index_value<PointType&, long> pt)
      { return pt.value().timestamp; };
  } else if (this->sensor == liom::SensorType::LIVOX) {
    point_time_cmp = [](const PointType& p1, const PointType& p2)
      { return p1.timestamp < p2.timestamp; };
    point_time_neq = [](boost::range::index_value<PointType&, long> p1,
                        boost::range::index_value<PointType&, long> p2)
      { return p1.value().timestamp != p2.value().timestamp; };
    extract_point_time = [&sweep_ref_time](boost::range::index_value<PointType&, long> pt)
      { return pt.value().timestamp * 1e-9f; };
  }

  // copy points into deskewed_scan_ in order of timestamp
  std::partial_sort_copy(this->original_scan->points.begin(), this->original_scan->points.end(),
                         deskewed_scan_->points.begin(), deskewed_scan_->points.end(), point_time_cmp);

  // filter unique timestamps
  auto points_unique_timestamps = deskewed_scan_->points
                                  | boost::adaptors::indexed()
                                  | boost::adaptors::adjacent_filtered(point_time_neq);

  // extract timestamps from points and put them in their own list
  std::vector<double> timestamps;
  std::vector<int> unique_time_indices;

  // compute offset between sweep reference time and first point timestamp
  double offset = 0.0;
  if (this->time_offset_) {
    offset = sweep_ref_time - extract_point_time(*points_unique_timestamps.begin());
  }

  // build list of unique timestamps and indices of first point with each timestamp
  for (auto it = points_unique_timestamps.begin(); it != points_unique_timestamps.end(); it++) {
    timestamps.push_back(extract_point_time(*it) + offset);
    unique_time_indices.push_back(it->index());
  }
  unique_time_indices.push_back(deskewed_scan_->points.size());

  int median_pt_index = timestamps.size() / 2;
  this->scan_stamp = timestamps[median_pt_index]; // set this->scan_stamp to the timestamp of the median point

  // don't process scans until IMU data is present
  if (!this->first_valid_scan) {
    if (this->imu_buffer.empty() || this->scan_stamp <= this->imu_buffer.back().stamp) {
      return;
    }

    this->first_valid_scan = true;
    this->T_prior = this->T; // assume no motion for the first scan
    pcl::transformPointCloud (*deskewed_scan_, *deskewed_scan_, this->T_prior * this->extrinsics.baselink2lidar_T);
    this->deskewed_scan = deskewed_scan_;
    this->deskew_status = true;
    return;
  }

  // IMU prior & deskewing for second scan onwards
  std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>> frames;

  Eigen::Vector3f prior_trans = this->lidarPose.p;
  this->timestampsTest = timestamps;
  frames = this->integrateImu(this->prev_scan_stamp, this->lidarPose.q, this->lidarPose.p,
                              this->geo.prev_vel.cast<float>(), timestamps);
  this->deskew_size = frames.size(); // if integration successful, equal to timestamps.size()

  // if there are no frames between the start and end of the sweep
  // that probably means that there's a sync issue
  if (frames.size() != timestamps.size()) {
    RCLCPP_FATAL(this->get_logger(),"Bad time sync between LiDAR and IMU!");

    this->T_prior = this->T;
    pcl::transformPointCloud (*deskewed_scan_, *deskewed_scan_, this->T_prior * this->extrinsics.baselink2lidar_T);
    this->deskewed_scan = deskewed_scan_;
    this->deskew_status = false;
    return;
  }

  // update prior to be the estimated pose at the median time of the scan (corresponds to this->scan_stamp)
  this->T_prior = frames[median_pt_index];

#pragma omp parallel for num_threads(this->num_threads_)
  for (int i = 0; i < timestamps.size(); i++) {

    Eigen::Matrix4f T = frames[i] * this->extrinsics.baselink2lidar_T;

    // transform point to world frame
    for (int k = unique_time_indices[i]; k < unique_time_indices[i+1]; k++) {
      auto &pt = deskewed_scan_->points[k];
      pt.getVector4fMap()[3] = 1.;
      pt.getVector4fMap() = T * pt.getVector4fMap();
    }
  }

  this->deskewed_scan = deskewed_scan_;
  this->deskew_status = true;

}

void liom::OdomNode::initializeInputTarget() {

  this->prev_scan_stamp = this->scan_stamp;

  // keep history of keyframes
  this->keyframes.push_back(std::make_pair(std::make_pair(this->lidarPose.p, this->lidarPose.q), this->current_scan));
  this->keyframe_timestamps.push_back(this->scan_header_stamp);
  this->keyframe_normals.push_back(this->gicp.getSourceCovariances());
  this->keyframe_transformations.push_back(this->T_corr);

}

void liom::OdomNode::setInputSource() {
  this->gicp.setInputSource(this->current_scan);
  this->gicp.calculateSourceCovariances();
}

void liom::OdomNode::initializeDLIO() {

  // Wait for IMU
  if (!this->first_imu_received || !this->imu_calibrated) {
    return;
  }

  this->liom_initialized = true;
  std::cout << std::endl << " DLIO initialized!" << std::endl;

}


/*
  Image callback
*/
void liom::OdomNode::callbackImage(const sensor_msgs::msg::Image::SharedPtr img) {
  // RCLCPP_ERROR(this->get_logger(), "Enter to Image node");
  try {
    std::unique_lock<std::mutex> lock(this->mtx_img);

    // Validate message
    if (!img){
      RCLCPP_WARN(this->get_logger(), "Received null image pointer");
      return;
    }

    double msg_header_time;
    try {
      msg_header_time = rclcpp::Time(img->header.stamp).seconds();
    } catch (...) {
      RCLCPP_ERROR(this->get_logger(), "Invalid header timestamp");
      return;
    }

    // Check for time consistency
    if (abs(msg_header_time - this->last_timestamp_img) < 0.001) return;
    if (msg_header_time < this->last_timestamp_img) {
      RCLCPP_ERROR(this->get_logger(), "image loop back.");
      RCLCPP_ERROR(this->get_logger(), "|message header time: %f |last image time: %f", 
                  msg_header_time, this->last_timestamp_img);
      return;
    }

    // Check time jump
    if (msg_header_time - last_timestamp_img < 0.02) {
      RCLCPP_WARN(this->get_logger(), "Image need Jumps: %6f", msg_header_time);
      return;
    }

    if (!this->liom_initialized) {
      return;
    }

    // Convert and store image
    cv::Mat img_curr;
    cv_bridge::CvImagePtr cv_ptr;
    try {
      img_curr = getImageFromMsg(img);
    } catch (...) {
      RCLCPP_ERROR(this->get_logger(), "Failed to convert image message");
      return;
    }

    // Limit buffer size 
    if (this->img_buffer_.size() > 50) {
      // RCLCPP_WARN(this->get_logger(), "Enter Image buffer");
      // std::cout << "|Size of the image buffer : " << this->img_buffer_.size() << std::endl;
      this->img_buffer_.pop_front();
      this->img_time_buffer_.pop_front();
    }
    this->img_buffer_.push_back(img_curr);
    this->img_time_buffer_.push_back(msg_header_time);
    last_timestamp_img = msg_header_time;

    try {
      cv_ptr   = cv_bridge::toCvCopy(img, sensor_msgs::image_encodings::BGR8);
    } catch (cv_bridge::Exception& e) {
      RCLCPP_ERROR(this->get_logger(), "Failed to convert image image bridge");
      return;
    }

    // Initialize for optical flow
    cv::Mat curr_frame, curr_frame_color;
    cv::cvtColor(cv_ptr->image, curr_frame, cv::COLOR_BGR2GRAY);
    cv_ptr->image.copyTo(curr_frame_color);
    
    // Compute optical flow
    if (!this->prev_frame.empty() && this->rgb_colorized) {
      std::vector<cv::Point2f> curr_pts;
      std::vector<uchar> status;
      std::vector<float> error;

      // Detect the feature in the previous frame
      cv::goodFeaturesToTrack(this->prev_opt_frame, this->prev_pts, 4500, 0.01, 10);
      cv::TermCriteria criteria = cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 0.01);
      
      // Compute the optical flow
      // cv::calcOpticalFlowPyrLK(this->prev_frame, curr_frame, this->prev_pts, curr_pts, status, error, cv::Size(21, 21), 3, criteria);
      cv::calcOpticalFlowPyrLK(this->prev_opt_frame, curr_frame, this->prev_pts, curr_pts, status, error, cv::Size(21, 21), 3, criteria);
      
      // RANSAC Outlier Filtering
      std::vector<cv::Point2f> prev_pts_filtered, curr_pts_filtered, prev_pts_filtered_, curr_pts_filtered_;
      std::vector<uchar> ransac_mask;

      // Filter out points with failed flow
      for (size_t i = 0; i < status.size(); i++) {
          if (status[i]) {
              prev_pts_filtered.push_back(prev_pts[i]);
              curr_pts_filtered.push_back(curr_pts[i]);
          }
      }

      // Apply RANSAC to find inlier and outlier
      if (prev_pts_filtered.size() >= 5) {
          cv::Mat F = cv::findFundamentalMat(
                  prev_pts_filtered, curr_pts_filtered,
                  cv::RANSAC, 0.3, 0.99, ransac_mask
          );
      }

      // Draw only inliers (green) and outliers (red)
      for (size_t i = 0; i < ransac_mask.size(); i++) {
          if (ransac_mask[i]) {  // Inlier
              cv::line(curr_frame_color, prev_pts_filtered[i], curr_pts_filtered[i], 
                      cv::Scalar(0, 255, 0), 1);  // Green line
              cv::circle(curr_frame_color, curr_pts_filtered[i], 3, 
                          cv::Scalar(0, 255, 255), -1);  // yellow dot
              // std::cout << "!Inlier Curr Points : = " << curr_pts_filtered[i] << "|Inlier Prev Points : = " << prev_pts_filtered[i] << std::endl; 
              prev_pts_filtered_.push_back(prev_pts_filtered[i]);
              curr_pts_filtered_.push_back(curr_pts_filtered[i]);
          } else {  // Outlier
              cv::circle(curr_frame_color, curr_pts_filtered[i], 3, 
                          cv::Scalar(0, 0, 255), -1);  // Red dot
          }
      }

      // Match the point cloud to the feature points
      this->greedyMatching(this->prev_projected_pts, prev_pts_filtered, 3, this->matches);
      if (this->visual_initialization && this->valid_colorized_points->size() > 0 && this->matches.size()){
          // Create filtered cloud and store the original indices
          pcl::PointCloud<pcl::PointXYZRGB>::Ptr filtered_cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZRGB>>();

          // RCLCPP_INFO(this->get_logger(), "Enter to Initial node");
          std::vector<int> cloud_idxs, frame_idxs, dist_;
          filtered_cloud->reserve(this->matches.size());
          cloud_idxs.reserve(this->matches.size());
          dist_.reserve(this->matches.size());
          frame_idxs.reserve(this->matches.size());

          for (const auto& m : this->matches) {
            filtered_cloud->push_back(this->prev_colorized_points->points[m.queryIdx]);
            dist_.push_back(m.distance);
            cloud_idxs.push_back(m.queryIdx);
            frame_idxs.push_back(m.trainIdx);
          }

          // Calculate the total observation obtical flow
          float total_dist = 0.0f;
          int valid_pts = 0;

          // RCLCPP_INFO(this->get_logger(), "Enter calculation node");
          for (const auto& idx : frame_idxs) {
            if (idx < prev_pts_filtered.size()) {  // Ensure the index is valid
                float dx = curr_pts_filtered[idx].x - prev_pts_filtered[idx].x;
                float dy = curr_pts_filtered[idx].y - prev_pts_filtered[idx].y;
                float distance = std::sqrt(dx * dx + dy * dy);  // Euclidean distance
                
                // flow_distances.push_back(distance);
                total_dist += distance;
                valid_pts++;
            }
          }

          // Create output cloud as shared pointer
          pcl::PointCloud<pcl::PointXYZRGB>::Ptr transformed_cloud(new pcl::PointCloud<pcl::PointXYZRGB>());
          // Transform the cloud - pass dereferenced pointers
          pcl::transformPointCloud(*filtered_cloud, *transformed_cloud, 
                                    Eigen::Affine3f(this->T_corr));

          // Find the corresponding optical flows                                  
          std::vector<int> correspondences;
          this->findCorrespondences(transformed_cloud, this->valid_colorized_points, correspondences, cloud_idxs, dist_);

          if (this->optical_est > total_dist) {
            RCLCPP_WARN(this->get_logger(), "Find the optical flow correspondence");
          }

          std::vector<int> corr;
          // RCLCPP_INFO(this->get_logger(), "Start optimization node");
          if (!this->prev_colorized_points || !this->valid_colorized_points || this->prev_colorized_points->empty() || this->valid_colorized_points->empty()) {
            RCLCPP_WARN(this->get_logger(), "Skipping optimization: Empty input clouds");
            return;
          }
          pcl::PointCloud<pcl::Normal>::Ptr target_normals = this->computeNormals(this->prev_colorized_points);
          this->optimization(this->prev_colorized_points, this->valid_colorized_points, transformed_cloud, target_normals, prev_pts_filtered, curr_pts_filtered, frame_idxs, cloud_idxs, this->T_corr, total_dist);
        }
      // Check optical flow status
      int num_success = std::count(status.begin(), status.end(), 1);

      // Update the previous frame
      this->prev_opt_frame = curr_frame;
      this->rgb_colorized  = false;
    }

    // Publish image with optical flow
    this->prev_frame = curr_frame;
    cv_bridge::CvImage cv_image;
    cv_image.image    = curr_frame_color.clone();
    cv_image.encoding = sensor_msgs::image_encodings::BGR8;
    cv_image.header.frame_id = "camera_frame";
    this->pubImage.publish(cv_image.toImageMsg());
  } catch (const std::exception& e) {
    RCLCPP_FATAL(this->get_logger(), "Exception in image callback: %s", e.what());
  } catch (...) {
    RCLCPP_FATAL(this->get_logger(), "Unknown exception in image callback");
  }
}

/*
    Define motion prior cost
*/
liom::OdomNode::MotionPriorCost::MotionPriorCost (const double* prior_pose, double weight)
                                                  : prior_pose_(prior_pose), weight_(weight) {}

/*
  Define the optical flow cost function
*/
liom::OdomNode::FlowCostFunction::FlowCostFunction (const std::vector<cv::Point2f> &curr_pts_filtered,
                     const std::vector<cv::Point2f> &prev_pts_filtered,
                     const std::vector<int> &frame_idxs,
                     const std::vector<int> &cloud_idxs,
                     const pcl::PointCloud<pcl::PointXYZRGB>::Ptr &source_cloud,
                     const pcl::PointCloud<pcl::PointXYZRGB>::Ptr &target_cloud,
                     const Eigen::Matrix3f& R_cl,
                     const Eigen::Vector3f& P_cl,
                     const Eigen::Matrix3f& R_corr,
                     const Eigen::Vector3f& P_corr,
                     const Eigen::Matrix3f &K,
                     double max_correspondence_dist = 0.5)
                     : curr_pts_(curr_pts_filtered),
                       prev_pts_(prev_pts_filtered),
                       frame_idxs_(frame_idxs),
                       cloud_idxs_(cloud_idxs),
                       source_cloud_(source_cloud),
                       target_cloud_(target_cloud),
                       R_cl_(R_cl), P_cl_(P_cl),
                       R_corr_(R_corr), P_corr_(P_corr),
                       K_(K),
                       max_correspondence_dist_(max_correspondence_dist) {
                       
                       if (!target_cloud_) {
                        throw std::invalid_argument("Target cloud is null");
                       }
                       // Pre-build KD-Tree for target cloud
                       kdtree_.setInputCloud(target_cloud);
}

/*
    Define Point To Point cost function
*/
liom::OdomNode::PointToPointCostFunction::PointToPointCostFunction (const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& source,
    const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& target,
    double max_dist)
    : source_(source), 
      target_(target), 
      max_corr_dist_sq_p2p_(max_dist * max_dist) {
      establishCorrespondencesP2P();
}

void liom::OdomNode::PointToPointCostFunction::establishCorrespondencesP2P() {
    pcl::KdTreeFLANN<pcl::PointXYZRGB> kdtree;
    kdtree.setInputCloud(target_);
    
    correspondences_.resize(source_->size());
    std::vector<int> indices(1);
    std::vector<float> distances(1);

    for (size_t i = 0; i < source_->size(); ++i) {
        if (kdtree.nearestKSearch(source_->points[i], 1, indices, distances) > 0) {
            if (distances[0] <= max_corr_dist_sq_p2p_) {
                correspondences_[i] = indices[0];
            } else {
                correspondences_[i] = -1;  // Mark as invalid
            }
        }
    }
}

/*
  Define GICP cost function
*/ 
liom::OdomNode::GICPCostFunction::GICPCostFunction (    const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& source_cloud,
    const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& target_cloud,
    const pcl::PointCloud<pcl::Normal>::Ptr& target_normals,
    double max_corr_dist)
    : source_cloud_(source_cloud),
      target_cloud_(target_cloud),
      target_normals_(target_normals),
      max_corr_dist_sq_(max_corr_dist * max_corr_dist) {
    establishCorrespondences();
}

void liom::OdomNode::GICPCostFunction::establishCorrespondences() {
    correspondences_.clear();
    pcl::KdTreeFLANN<pcl::PointXYZRGB> kdtree;
    kdtree.setInputCloud(target_cloud_);

    for (size_t i = 0; i < source_cloud_->size(); ++i) {
        const auto& point = source_cloud_->points[i];
        if (!pcl::isFinite(point)) continue;
        
        std::vector<int> indices(1);
        std::vector<float> distances(1);
        
        if (kdtree.nearestKSearch(point, 1, indices, distances) > 0) {
            if (distances[0] <= max_corr_dist_sq_) {
                // Additional check for valid normal
                if (pcl::isFinite(target_normals_->points[indices[0]])) {
                    correspondences_.push_back(indices[0]);
                }
            }
        }
    }
}


/*
  Optimization function
*/
void liom::OdomNode::optimization(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr &source_cloud,
                                  const pcl::PointCloud<pcl::PointXYZRGB>::Ptr &target_cloud,
                                  const pcl::PointCloud<pcl::PointXYZRGB>::Ptr &transformed_cloud,
                                  const pcl::PointCloud<pcl::Normal>::Ptr &target_normals,
                                  const std::vector<cv::Point2f> &prev_pts,
                                  const std::vector<cv::Point2f> &curr_pts,
                                  const std::vector<int> &frame_idxs,
                                  const std::vector<int> &cloud_idxs,
                                  const Eigen::Matrix4f &T_corr,
                                  int total_dist) {

  // Input Validation                                     
  if (!source_cloud || !target_cloud || source_cloud->empty() || target_cloud->empty()) {
    RCLCPP_WARN(this->get_logger(), "Skipping optimization: Empty input clouds");
    return;
  }
                                    
  // Convert quaternion to angle-axis for Ceres
  Eigen::AngleAxisf init_rotation(this->state.q);
  double pose[6] = {
      init_rotation.angle() * init_rotation.axis().x(),
      init_rotation.angle() * init_rotation.axis().y(),
      init_rotation.angle() * init_rotation.axis().z(),
      this->state.p.x(),  // position from state
      this->state.p.y(),
      this->state.p.z()
  };

  // Store prior motion
  double prior_pose[6];
  std::copy(pose, pose+6, prior_pose);

  // Extract the correlation transform
  Eigen::Matrix3f R_corr = T_corr.block<3, 3>(0, 0);
  Eigen::Vector3f P_corr = T_corr.block<3, 1>(0, 3);

  // Weight for optimization
  const double gicp_weight = 1.0;
  const double p2p_weight  = 1.0;
  const double flow_weight = 0.01;
  const double motion_prior_weight = 0.8;
  // const double motion_prior_weight = 10;

  ceres::Problem problem;
  bool valid_residuals = false;
  
  // Add GICP residuals
  // auto* gicp_cost = new GICPCostFunction(source_cloud, target_cloud, target_normals, this->gicp_max_corr_dist_);
  auto* gicp_cost = new GICPCostFunction(target_cloud, source_cloud, target_normals, this->gicp_max_corr_dist_);
  const size_t gicp_correspondences = gicp_cost->getNumCorrespondences();

  problem.AddResidualBlock(
    new ceres::AutoDiffCostFunction<MotionPriorCost, 6, 6>(
        new MotionPriorCost(prior_pose, motion_prior_weight)),
    nullptr,
    pose
  );

  auto flow_cost = std::make_unique<FlowCostFunction>(
      curr_pts, prev_pts, frame_idxs, cloud_idxs,
      source_cloud, target_cloud,
      this->Rcl, this->Pcl,
      R_corr, P_corr,
      this->camera.intrinsic_matrix,
      0.5);

  std::vector<double> test_residuals(2 * frame_idxs.size());
  if ((*flow_cost)(pose, test_residuals.data())) {
    int residual_count = flow_cost->getValidCount();
    // RCLCPP_INFO(this->get_logger(), "Successfully added flow residuals (%d points)", residual_count);
    if (residual_count > 50) {
      RCLCPP_INFO(this->get_logger(), "Add Flow cost");
      problem.AddResidualBlock(
          new ceres::AutoDiffCostFunction<FlowCostFunction, ceres::DYNAMIC, 6>(
              flow_cost.release(), 2 * frame_idxs.size()),
          new ceres::ScaledLoss(
              new ceres::CauchyLoss(0.5),
              flow_weight,
              ceres::TAKE_OWNERSHIP),
          pose
      );
      // RCLCPP_WARN(this->get_logger(), "Ends AddResidualBlock");
      RCLCPP_INFO(this->get_logger(), "Successfully added flow residuals (%d points)", residual_count);
    } else {
      RCLCPP_WARN(this->get_logger(), "Skipped flow residuals: insufficient valid correspondences (%d)", residual_count);
    }
  } else {
    RCLCPP_WARN(this->get_logger(), "Flow residuals rejected during pre-evaluation");
  }

  // Configure Solver
  ceres::Solver::Options options;
  options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;  // Best for large problems
  options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
  options.dogleg_type = ceres::TRADITIONAL_DOGLEG;
  options.num_threads = 2;
  options.max_num_iterations = 20;
  options.function_tolerance = 1e-6;
  options.parameter_tolerance = 1e-8;
  options.minimizer_progress_to_stdout = true;  // Debug output
  options.num_threads = std::thread::hardware_concurrency();

  // Trust region strategy (critical for hybrid residuals)
  // options.minimizer_progress_to_stdout = true;
  options.update_state_every_iteration = true;


// Check all parameter blocks
  std::vector<double*> parameter_blocks;
  problem.GetParameterBlocks(&parameter_blocks);
  for (auto* block : parameter_blocks) {
      for (int i = 0; i < 6; ++i) {
        if (!std::isfinite(block[i])) {
          RCLCPP_ERROR(this->get_logger(), "Invalid parameter at position %d: %f", i, block[i]);
        }
      }
  }
  // Solve
  // RCLCPP_INFO(this->get_logger(), "Solve optimization");
  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);
  // RCLCPP_WARN(this->get_logger(), "Ends Solve");

  if (!summary.IsSolutionUsable()) {
    RCLCPP_ERROR(this->get_logger(), "Optimization failed: %s", 
                  summary.message.c_str());
  }
  // Print full report
  // std::cout << summary.FullReport() << std::endl;
  RCLCPP_INFO_STREAM(this->get_logger(), summary.BriefReport());
  // std::cout << summary.FullReport() << "\n";

  // Convert optimized Ceres pose back to state
  Eigen::Vector3d angle_axis(pose[0], pose[1], pose[2]);
  double angle = angle_axis.norm();

  if (angle > 1e-10) {
      Eigen::Vector3d axis = angle_axis / angle;
      Eigen::Quaternionf q = Eigen::Quaternionf(
          Eigen::AngleAxisf(static_cast<float>(angle), 
                          axis.cast<float>()));
      this->state.q = q;
  } else {
      this->state.q = Eigen::Quaternionf::Identity();
  }

  Eigen::Vector3f p = Eigen::Vector3f(
      static_cast<float>(pose[3]),
      static_cast<float>(pose[4]),
      static_cast<float>(pose[5]));
      this->state.p = 0.8 * this->state.p + 0.2 * p;
}



/*
  Compute target normals
*/
pcl::PointCloud<pcl::Normal>::Ptr liom::OdomNode::computeNormals(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr &cloud) {
  pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
  pcl::NormalEstimation<pcl::PointXYZRGB, pcl::Normal> ne;
  ne.setInputCloud(cloud);
  pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZRGB>());
  ne.setSearchMethod(tree);

  // Use KNN (K-nearest neighbor)
  const int k = 5; 
  ne.setKSearch(k);
  ne.compute(*normals);

  // ne.setRadiusSearch(0.2);
  // ne.compute(*normals);

  for (auto& normal : *normals) {
    if (!pcl::isFinite(normal)) {
      normal.normal_x = normal.normal_y = normal.normal_z = 0;
      normal.curvature = 0;
    }
  }
  return normals;
}

/*
  Calculate flow cost of the estimation
*/
void liom::OdomNode::calcFlowEstCost(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr &src_cloud,
                       const pcl::PointCloud<pcl::PointXYZRGB>::Ptr &tgt_cloud,
                       const std::vector<cv::Point2f> &curr_pts,
                       const std::vector<cv::Point2f> &prev_pts,
                       const std::vector<int> &frame_idxs) {
  
  // Input validation
  if (!src_cloud || src_cloud->empty() || !tgt_cloud || tgt_cloud->empty()) {
    RCLCPP_ERROR(this->get_logger(), "Invalid input clouds");
    return;
  }
  pcl::KdTreeFLANN<pcl::PointXYZRGB> kdtree;
  kdtree.setInputCloud(tgt_cloud);
  std::vector<int> indices(1);
  std::vector<float> distances(1);

  Eigen::Matrix3f R_wi = this->state.rot;                                      // World2imu
  Eigen::Vector3f P_wi = this->state.p;   
  Eigen::Matrix3f R_cw = this->Rcl * R_wi.transpose();                        // Camera2world
  Eigen::Vector3f P_cw = - this->Rcl* R_wi.transpose() * P_wi + this->Pcl;

  float opt_est = 0.0f;

  for (size_t i = 0; i < src_cloud->size(); i++) {
      // Skip NaN points
      if (!pcl::isFinite(src_cloud->points[i])) {
          continue;
      }

      if (kdtree.nearestKSearch(src_cloud->points[i], 1, indices, distances) == 0) {
          continue;  // No neighbors found
      }

      if (distances[0] < 0.2 && indices[0] >= 0 && indices[0] < tgt_cloud->size()) {
        int tgt_idx = indices[0];
        if (tgt_idx < 0 || tgt_idx >= tgt_cloud->size()) continue;
        if (!pcl::isFinite(tgt_cloud->points[tgt_idx])) continue;

        // Validate frame index
        int obs_idx;
        try {
              obs_idx = frame_idxs.at(i);
              if (obs_idx < 0 || obs_idx >= prev_pts.size() || obs_idx >= curr_pts.size()) {
                  continue;
              }
        } catch (const std::out_of_range& e) {
              continue;
        }
        
        Eigen::Vector3f pt(tgt_cloud->points[tgt_idx].x, 
                           tgt_cloud->points[tgt_idx].y,
                           tgt_cloud->points[tgt_idx].z);
        Eigen::Vector3f   pc =  R_cw * pt + P_cw;
        Eigen::Vector3f image_points = this->camera.intrinsic_matrix * pc;            
        
        // Normalize to get pixel coordinates
        float u = image_points(0) / image_points(2);
        float v = image_points(1) / image_points(2);

        float dx = u - prev_pts[obs_idx].x;
        float dy = v - prev_pts[obs_idx].y;
        opt_est += std::sqrt(dx * dx + dy * dy);
      }
  }
  this->optical_est = opt_est;
}

/*
  Find correspondences function
*/
void liom::OdomNode::findCorrespondences(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr &source_cloud,
                                         const pcl::PointCloud<pcl::PointXYZRGB>::Ptr &target_cloud,
                                         std::vector<int> &correspondences,
                                         const std::vector<int> &cloud_idxs,
                                         const std::vector<int> &dist_) {

  // Input validation
  if (!source_cloud || source_cloud->empty() || !target_cloud || target_cloud->empty()) {
    RCLCPP_ERROR(this->get_logger(), "Invalid input clouds");
    correspondences.clear();
    return;
  }

  if (!cloud_idxs.empty() && cloud_idxs.size() != source_cloud->size()) {
    RCLCPP_ERROR(this->get_logger(), "cloud_idxs size mismatch with source cloud");
    correspondences.clear();
    return;
  }


  pcl::KdTreeFLANN<pcl::PointXYZRGB> kdtree;
  kdtree.setInputCloud(target_cloud);
  std::vector<int> indices(1);
  std::vector<float> distances(1);

  correspondences.resize(source_cloud->size(), -1);
  int count = 0;
  int count_opt = 0;
  
  float opt_est = 0.0f;
  float opt_obs = 0.0f;

  for (size_t i = 0; i < source_cloud->size(); ++i) {

      // Skip NaN points
      if (!pcl::isFinite(source_cloud->points[i])) {
          continue;
      }

      /*
        If Option 2 uncomment these lines below:
      */
      if (kdtree.nearestKSearch(source_cloud->points[i], 1, indices, distances) == 0) {
          continue;  // No neighbors found
      }

      // Validate the found index
      if (distances[0] < 0.2 && indices[0] >= 0 && indices[0] < target_cloud->size()) {
          correspondences[i] = indices[0];
          const auto& src_pt = source_cloud->points[i];
          const auto& tgt_pt = target_cloud->points[indices[0]];
          
          int src_idx = cloud_idxs.empty() ? i : cloud_idxs[i];
          // RCLCPP_INFO(this->get_logger(), 
          //             "%8d %8d (%6.3f,%6.3f,%6.3f | %3d,%3d,%3d) (%6.3f,%6.3f,%6.3f | %3d,%3d,%3d) %8.3f",
          //             src_idx, indices[0],
          //             src_pt.x, src_pt.y, src_pt.z,
          //             src_pt.r, src_pt.g, src_pt.b,
          //             tgt_pt.x, tgt_pt.y, tgt_pt.z,
          //             tgt_pt.r, tgt_pt.g, tgt_pt.b,
          //             distances[0]);
          count++;

          // Extract the corresponding pixels between two frames
          if (src_idx >= 0 && src_idx < this->prev_projected_pts.size() && 
              indices[0] >= 0 && indices[0] < this->projected_pts.size()) {
              // std::cout << "|Prev projected pixels : " << this->prev_projected_pts[src_idx] 
                        // << "|Curr projected pixels : " << this->projected_pts[indices[0]] << std::endl;
              float dx = this->prev_projected_pts[src_idx].x - this->projected_pts[indices[0]].x;
              float dy = this->prev_projected_pts[src_idx].y - this->projected_pts[indices[0]].y;
              opt_est += std::sqrt(dx * dx + dy * dy);
              // std::cout << " |Optical distance estimate : " << opt_est << std::endl;
              count_opt++;
          } else {
              RCLCPP_WARN(this->get_logger(), "Projected points index out of bounds (src: %d/%zu, tgt: %d/%zu)",
                          src_idx, this->prev_projected_pts.size(), 
                          indices[0], this->projected_pts.size());
          }         
      }
      opt_obs += dist_[i];
  }
  this->optical_est = opt_est;
}

/*
  Matching Point cloud to image features
*/
void liom::OdomNode::greedyMatching(const std::vector<cv::Point2f> &projected_pts, 
                                    const std::vector<cv::Point2f> &featured_pts,
                                    float max_dist_thres,
                                    std::vector<cv::DMatch> &matches) {
  // RCLCPP_ERROR(this->get_logger(), "Enter to matching node");
  // this->matches.clear();
  matches.clear();
  std::vector<bool> feature_matched(featured_pts.size(), false);

  for (size_t i = 0; i < projected_pts.size(); ++i) {
    float min_dist = std::numeric_limits<float>::max();
    int best_j = -1;

    for (size_t j = 0; j < featured_pts.size(); ++j) {
      if (feature_matched[j]) continue;

      float dx = projected_pts[i].x - featured_pts[j].x;
      float dy = projected_pts[i].y - featured_pts[j].y;
      float dist = std::sqrt(dx * dx + dy * dy);

      if (dist < min_dist && dist < max_dist_thres) {
        min_dist = dist;
        best_j   = j;
      }
    }

    if (best_j != -1) {
      this->matches.emplace_back(i, best_j, min_dist);
      feature_matched[best_j] = true;
    }
  }

  int count = 0;
}

/*
  Sub-Function for Image callback
*/
cv::Mat liom::OdomNode::getImageFromMsg(const sensor_msgs::msg::Image::ConstSharedPtr &img_msg)
{
    cv::Mat img;
    try {
        // Convert ROS2 image message to OpenCV Mat
        img = cv_bridge::toCvCopy(img_msg, "bgr8")->image;
    } catch (const cv_bridge::Exception& e) {
        RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
        return cv::Mat();  // Return empty Mat on error
    }
    return img;
}

/*
  Projecting LiDAR point to image
*/
void liom::OdomNode::projectLidarToImage(const cv::Mat &img) {
  
  // 1. Initialize PCL point clouds
  pcl::PointCloud<PointType>::Ptr voxelized_cloud = std::make_shared<pcl::PointCloud<PointType>>();
  pcl::PointCloud<PointType>::Ptr deskewed_scan_t_visual = std::make_shared<pcl::PointCloud<PointType>>();
  pcl::transformPointCloud(*this->deskewed_scan, *deskewed_scan_t_visual, this->T_corr );

  // 2. Voxel Grid Filtering
  pcl::VoxelGrid<PointType> voxel_filter;
  voxel_filter.setInputCloud(deskewed_scan_t_visual);     // Input cloud (transformed)
  voxel_filter.setLeafSize(0.001f, 0.001f, 0.001f);       // Voxel size (adjust as needed)
  voxel_filter.filter(*voxelized_cloud);                  // Output: voxelized_cloud
  // RCLCPP_DEBUG(this->get_logger(), "Voxelized cloud size: %zu", voxelized_cloud->size());
  this->valid_colorized_points = std::make_shared<pcl::PointCloud<pcl::PointXYZRGB>>();
  this->valid_colorized_points->points.reserve(voxelized_cloud->size());
  this->prev_colorized_points = std::make_shared<pcl::PointCloud<pcl::PointXYZRGB>>();
  this->prev_colorized_points->points.reserve(voxelized_cloud->size());
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr laserCloudWorldRGB = std::make_shared<pcl::PointCloud<pcl::PointXYZRGB>>();
  laserCloudWorldRGB->reserve(voxelized_cloud->size());

  // Exrtact the state orientation
  Eigen::Quaternionf q = this->state.q.normalized();
  this->state.rot      = q.toRotationMatrix();
  // Calculate the transformation between coordinates
  Eigen::Matrix3f R_wi = this->state.rot;                                      // World2imu
  Eigen::Vector3f P_wi = this->state.p;   
  Eigen::Matrix3f R_cw = this->Rcl * R_wi.transpose();                        // Camera2world
  Eigen::Vector3f P_cw = - this->Rcl* R_wi.transpose() * P_wi + this->Pcl;

  if (img.empty()) {
  RCLCPP_WARN(this->get_logger(), "Image is empty, skipping projection.");
  return;
  }

  this->projected_pts.clear();                                                // Clear the previous points
  this->projected_pts.reserve(voxelized_cloud->size());                       // Pre-allocate memory
  this->prev_projected_pts.reserve(voxelized_cloud->size());

  if (this->visual_initialization) {
    for (int i = 0; i < voxelized_cloud->size(); i++) {
    // for (int i = 0; i < this->deskewed_scan->size(); i++) {
        Eigen::Vector3f pt(voxelized_cloud->points[i].x,
              voxelized_cloud->points[i].y,
              voxelized_cloud->points[i].z);

        Eigen::Vector3f   pc =  R_cw * pt + P_cw;
        Eigen::Vector3f image_points = this->camera.intrinsic_matrix * pc;

        // Normalize to get pixel coordinates
        float u = image_points(0) / image_points(2);
        float v = image_points(1) / image_points(2);
        if (u < 0 || u >= this->camera.width || v < 0 || v >= this->camera.height || image_points(2) <= 0) {
            continue;
        }

        if (u >= 0 && u < (this->camera.width - 1) && v >= 0 && v < (this->camera.height - 1) && image_points(2) > 0) {
          // RCLCPP_WARN(this->get_logger(), "Valid image pointer");
          this->projected_pts.emplace_back(u, v);
          int u_floor = static_cast<int>(u);
          int v_floor = static_cast<int>(v);
          float u_frac = u - u_floor;
          float v_frac = v - v_floor;

          // Get the 4 neighboring pixels
          cv::Vec3b p00 = img.at<cv::Vec3b>(v_floor, u_floor);
          cv::Vec3b p01 = img.at<cv::Vec3b>(v_floor, u_floor + 1);
          cv::Vec3b p10 = img.at<cv::Vec3b>(v_floor + 1, u_floor);
          cv::Vec3b p11 = img.at<cv::Vec3b>(v_floor + 1, u_floor + 1);

          // Interpolate each channel (BGR) separately
          cv::Vec3b color;
          for (int i = 0; i < 3; i++) {
              float channel_val = 
                  (1 - u_frac) * (1 - v_frac) * p00[i] +
                  u_frac * (1 - v_frac) * p01[i] +
                  (1 - u_frac) * v_frac * p10[i] +
                  u_frac * v_frac * p11[i];
              color[i] = static_cast<unsigned char>(std::round(channel_val));
          }
          pcl::PointXYZRGB colored_point;
          colored_point.x = pt(0);
          colored_point.y = pt(1);
          colored_point.z = pt(2);
          colored_point.r = color[2];
          colored_point.g = color[1];
          colored_point.b = color[0];
          // // Add point to rgb point cloud
          this->valid_colorized_points->points.push_back(colored_point);
          laserCloudWorldRGB->push_back(colored_point);
        }
      }
      this->valid_point_flag = true;
      this->cloud_rgb_buffer.push_back(this->valid_colorized_points);
      this->pts_buffer.push_back(this->projected_pts);
  } else {
    for (int i = 0; i < voxelized_cloud->size(); i++) {
        Eigen::Vector3f pt(voxelized_cloud->points[i].x,
              voxelized_cloud->points[i].y,
              voxelized_cloud->points[i].z);

        Eigen::Vector3f   pc =  R_cw * pt + P_cw;
        Eigen::Vector3f image_points = this->camera.intrinsic_matrix * pc;

        // Normalize to get pixel coordinates
        float u = image_points(0) / image_points(2);
        float v = image_points(1) / image_points(2);
        
        if (u < 0 || u >= this->camera.width || v < 0 || v >= this->camera.height || image_points(2) <= 0) {
            continue;
        }

        if (u >= 0 && u < this->camera.width && v >= 0 && v < this->camera.height && image_points(2) > 0) {
          this->prev_projected_pts.emplace_back(u, v);
          cv::Vec3b color = img.at<cv::Vec3b>(v, u);

          pcl::PointXYZRGB colored_point;
          colored_point.x = pt(0);
          colored_point.y = pt(1);
          colored_point.z = pt(2);
          colored_point.r = color[2];
          colored_point.g = color[1];
          colored_point.b = color[0];
          // // Add point to rgb point cloud
          this->prev_colorized_points->points.push_back(colored_point);
          laserCloudWorldRGB->push_back(colored_point);
        }
    }
    this->prev_opt_frame = this->prev_frame;
  }

  if (laserCloudWorldRGB->size() > 0) {
    this->visual_initialization = true;
  }

  if (this->valid_colorized_points->size() > 0 && this->projected_pts.size() > 0) {
    this->prev_colorized_points = this->valid_colorized_points;
    this->prev_projected_pts    = this->projected_pts;
  }

  // pcl::PointCloud<pcl::PointXYZRGB>::Ptr colorized_pts_t = std::make_shared<pcl::PointCloud<pcl::PointXYZRGB>>();
  // sensor_msgs::msg::PointCloud2 rgb_pc;
  // pcl::toROSMsg(*laserCloudWorldRGB, rgb_pc);
  // rgb_pc.header.stamp = this->scan_header_stamp;
  // rgb_pc.header.frame_id = this->odom_frame;
  // this->deskewed_pub->publish(rgb_pc);
  this->rgb_colorized = true;
  // RCLCPP_INFO(this->get_logger(), "Number of  %zu voxel", voxelized_cloud->size());
  // RCLCPP_INFO(this->get_logger(), "Stored %zu projected points.", projected_pts.size());
}

/*
  PointCloud callback
*/
void liom::OdomNode::callbackPointCloud(const sensor_msgs::msg::PointCloud2::SharedPtr pc) {

  std::unique_lock<decltype(this->main_loop_running_mutex)> lock(main_loop_running_mutex);
  this->main_loop_running = true;
  lock.unlock();

  //   double then = this->now().seconds();
  rclcpp::Clock clock;
  double then = clock.now().seconds();

  if (this->first_scan_stamp == 0.) {
    this->first_scan_stamp = rclcpp::Time(pc->header.stamp).seconds();
  }

  // liom Initialization procedures (IMU calib, gravity align)
  if (!this->liom_initialized) {
    this->initializeDLIO();
  }

  // Convert incoming scan into DLIO format
  this->getScanFromROS(pc);

  // Preprocess points
  this->preprocessPoints();

  if (!this->first_valid_scan) {
    return;
  }

  if (this->current_scan->points.size() <= this->gicp_min_num_points_) {
    RCLCPP_FATAL(this->get_logger(), "Low number of points in the cloud!");
    return;
  }

  // Compute Metrics
  this->metrics_thread = std::thread( &liom::OdomNode::computeMetrics, this );
  this->metrics_thread.detach();

  // Set Adaptive Parameters
  if (this->adaptive_params_) {
    this->setAdaptiveParams();
  }

  // Set new frame as input source
  this->setInputSource();

  // Set initial frame as first keyframe
  if (this->keyframes.size() == 0) {
    this->initializeInputTarget();
    this->main_loop_running = false;
    this->submap_future =
      std::async( std::launch::async, &liom::OdomNode::buildKeyframesAndSubmap, this, this->state );
    this->submap_future.wait(); // wait until completion
    return;
  }

  // Get the next pose via IMU + S2M + GEO
  this->getNextPose();

  // Modification
  if (this->img_buffer_.size() > 1) {
    cv::Mat img = this->img_buffer_.back();
    this->projectLidarToImage(img);
  }

  // Update current keyframe poses and map
  this->updateKeyframes();

  // Build keyframe normals and submap if needed (and if we're not already waiting)
  if (this->new_submap_is_ready) {
    this->main_loop_running = false;
    this->submap_future =
      std::async( std::launch::async, &liom::OdomNode::buildKeyframesAndSubmap, this, this->state );
  } else {
    lock.lock();
    this->main_loop_running = false;
    lock.unlock();
    this->submap_build_cv.notify_one();
  }

  // Update trajectory
  this->trajectory.push_back( std::make_pair(this->state.p, this->state.q) );

  // Update time stamps
  this->lidar_rates.push_back( 1. / (this->scan_stamp - this->prev_scan_stamp) );
  this->prev_scan_stamp = this->scan_stamp;
  this->elapsed_time = this->scan_stamp - this->first_scan_stamp;

  // Publish stuff to ROS
  pcl::PointCloud<PointType>::ConstPtr published_cloud;
  if (this->densemap_filtered_) {
    published_cloud = this->current_scan;
  } else {
    published_cloud = this->deskewed_scan;
  }
  this->publish_thread = std::thread( &liom::OdomNode::publishToROS, this, published_cloud, this->T_corr );
  this->publish_thread.detach();

  // Update some statistics
//   this->comp_times.push_back(this->now().seconds() - then);
  this->comp_times.push_back(clock.now().seconds() - then);
  this->gicp_hasConverged = this->gicp.hasConverged();

  // Debug statements and publish custom liom message
  this->debug_thread = std::thread( &liom::OdomNode::debug, this );
  this->debug_thread.detach();

  this->geo.first_opt_done = true;

}

void liom::OdomNode::callbackImu(const sensor_msgs::msg::Imu::SharedPtr imu_raw) {

  this->first_imu_received = true;

  sensor_msgs::msg::Imu::SharedPtr imu = this->transformImu( imu_raw );
  this->imu_stamp = imu->header.stamp;
  double imu_stamp_secs = rclcpp::Time(imu->header.stamp).seconds();

  Eigen::Vector3f lin_accel;
  Eigen::Vector3f ang_vel;

  // Get IMU samples
  ang_vel[0] = imu->angular_velocity.x;
  ang_vel[1] = imu->angular_velocity.y;
  ang_vel[2] = imu->angular_velocity.z;

  lin_accel[0] = imu->linear_acceleration.x;
  lin_accel[1] = imu->linear_acceleration.y;
  lin_accel[2] = imu->linear_acceleration.z;

  if (this->first_imu_stamp == 0.) {
    this->first_imu_stamp = imu_stamp_secs;
  }

  // IMU calibration procedure - do for three seconds
  if (!this->imu_calibrated) {

    static int num_samples = 0;
    static Eigen::Vector3f gyro_avg (0., 0., 0.);
    static Eigen::Vector3f accel_avg (0., 0., 0.);
    static bool print = true;

    if ((imu_stamp_secs - this->first_imu_stamp) < this->imu_calib_time_) {

      num_samples++;

      gyro_avg[0] += ang_vel[0];
      gyro_avg[1] += ang_vel[1];
      gyro_avg[2] += ang_vel[2];

      accel_avg[0] += lin_accel[0];
      accel_avg[1] += lin_accel[1];
      accel_avg[2] += lin_accel[2];

      if(print) {
        std::cout << std::endl << " Calibrating IMU for " << this->imu_calib_time_ << " seconds... ";
        std::cout.flush();
        print = false;
      }

    } else {

      std::cout << "done" << std::endl << std::endl;

      gyro_avg /= num_samples;
      accel_avg /= num_samples;

      Eigen::Vector3f grav_vec (0., 0., this->gravity_);

      if (this->gravity_align_) {

        // Estimate gravity vector - Only approximate if biases have not been pre-calibrated
        grav_vec = (accel_avg - this->state.b.accel).normalized() * abs(this->gravity_);
        Eigen::Quaternionf grav_q = Eigen::Quaternionf::FromTwoVectors(grav_vec, Eigen::Vector3f(0., 0., this->gravity_));

        // set gravity aligned orientation
        this->state.q = grav_q;
        this->T.block(0,0,3,3) = this->state.q.toRotationMatrix();
        this->lidarPose.q = this->state.q;

        // rpy
        auto euler = grav_q.toRotationMatrix().eulerAngles(2, 1, 0);
        double yaw = euler[0] * (180.0/M_PI);
        double pitch = euler[1] * (180.0/M_PI);
        double roll = euler[2] * (180.0/M_PI);

        // use alternate representation if the yaw is smaller
        if (abs(remainder(yaw + 180.0, 360.0)) < abs(yaw)) {
          yaw   = remainder(yaw + 180.0,   360.0);
          pitch = remainder(180.0 - pitch, 360.0);
          roll  = remainder(roll + 180.0,  360.0);
        }
        std::cout << " Estimated initial attitude:" << std::endl;
        std::cout << "   Roll  [deg]: " << to_string_with_precision(roll, 4) << std::endl;
        std::cout << "   Pitch [deg]: " << to_string_with_precision(pitch, 4) << std::endl;
        std::cout << "   Yaw   [deg]: " << to_string_with_precision(yaw, 4) << std::endl;
        std::cout << std::endl;
      }

      if (this->calibrate_accel_) {

        // subtract gravity from avg accel to get bias
        this->state.b.accel = accel_avg - grav_vec;

        std::cout << " Accel biases [xyz]: " << to_string_with_precision(this->state.b.accel[0], 8) << ", "
                                             << to_string_with_precision(this->state.b.accel[1], 8) << ", "
                                             << to_string_with_precision(this->state.b.accel[2], 8) << std::endl;
      }

      if (this->calibrate_gyro_) {

        this->state.b.gyro = gyro_avg;

        std::cout << " Gyro biases  [xyz]: " << to_string_with_precision(this->state.b.gyro[0], 8) << ", "
                                             << to_string_with_precision(this->state.b.gyro[1], 8) << ", "
                                             << to_string_with_precision(this->state.b.gyro[2], 8) << std::endl;
      }

      this->imu_calibrated = true;

    }

  } else {

    double dt = imu_stamp_secs - this->prev_imu_stamp;
    if (dt == 0) { dt = 1.0/200.0; }
    this->imu_rates.push_back( 1./dt );

    // Apply the calibrated bias to the new IMU measurements
    this->imu_meas.stamp = imu_stamp_secs;
    this->imu_meas.dt = dt;
    this->prev_imu_stamp = this->imu_meas.stamp;

    Eigen::Vector3f lin_accel_corrected = (this->imu_accel_sm_ * lin_accel) - this->state.b.accel;
    Eigen::Vector3f ang_vel_corrected = ang_vel - this->state.b.gyro;

    this->imu_meas.lin_accel = lin_accel_corrected;
    this->imu_meas.ang_vel = ang_vel_corrected;

    // Store calibrated IMU measurements into imu buffer for manual integration later.
    this->mtx_imu.lock();
    this->imu_buffer.push_front(this->imu_meas);
    this->mtx_imu.unlock();

    // Notify the callbackPointCloud thread that IMU data exists for this time
    this->cv_imu_stamp.notify_one();

    if (this->geo.first_opt_done) {
      // Geometric Observer: Propagate State
      this->propagateState();
    }

  }

}

void liom::OdomNode::getNextPose() {

  // Check if the new submap is ready to be used
  this->new_submap_is_ready = (this->submap_future.wait_for(std::chrono::seconds(0)) == std::future_status::ready);

  if (this->new_submap_is_ready && this->submap_hasChanged) {

    // Set the current global submap as the target cloud
    this->gicp.registerInputTarget(this->submap_cloud);

    // Set submap kdtree
    this->gicp.target_kdtree_ = this->submap_kdtree;

    // Set target cloud's normals as submap normals
    this->gicp.setTargetCovariances(this->submap_normals);

    this->submap_hasChanged = false;
  }

  // Align with current submap with global IMU transformation as initial guess
  pcl::PointCloud<PointType>::Ptr aligned = std::make_shared<pcl::PointCloud<PointType>>();
  this->gicp.align(*aligned);

  // Get final transformation in global frame
  this->T_corr = this->gicp.getFinalTransformation(); // "correction" transformation
  this->T = this->T_corr * this->T_prior;

  // Update next global pose
  // Both source and target clouds are in the global frame now, so tranformation is global
  this->propagateGICP();

  // Geometric observer update
  this->updateState();

}

bool liom::OdomNode::imuMeasFromTimeRange(double start_time, double end_time,
                                          boost::circular_buffer<ImuMeas>::reverse_iterator& begin_imu_it,
                                          boost::circular_buffer<ImuMeas>::reverse_iterator& end_imu_it) {

  if (this->imu_buffer.empty() || this->imu_buffer.front().stamp < end_time) {
    // Wait for the latest IMU data
    std::unique_lock<decltype(this->mtx_imu)> lock(this->mtx_imu);
    this->cv_imu_stamp.wait(lock, [this, &end_time]{ return this->imu_buffer.front().stamp >= end_time; });
  }

  auto imu_it = this->imu_buffer.begin();

  auto last_imu_it = imu_it;
  imu_it++;
  while (imu_it != this->imu_buffer.end() && imu_it->stamp >= end_time) {
    last_imu_it = imu_it;
    imu_it++;
  }

  while (imu_it != this->imu_buffer.end() && imu_it->stamp >= start_time) {
    imu_it++;
  }

  if (imu_it == this->imu_buffer.end()) {
    // not enough IMU measurements, return false
    return false;
  }
  imu_it++;

  // Set reverse iterators (to iterate forward in time)
  end_imu_it = boost::circular_buffer<ImuMeas>::reverse_iterator(last_imu_it);
  begin_imu_it = boost::circular_buffer<ImuMeas>::reverse_iterator(imu_it);

  return true;
}

std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>>
liom::OdomNode::integrateImu(double start_time, Eigen::Quaternionf q_init, Eigen::Vector3f p_init,
                             Eigen::Vector3f v_init, const std::vector<double>& sorted_timestamps) {

  const std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>> empty;

  if (sorted_timestamps.empty() || start_time > sorted_timestamps.front()) {
    // invalid input, return empty vector
    return empty;
  }

  boost::circular_buffer<ImuMeas>::reverse_iterator begin_imu_it;
  boost::circular_buffer<ImuMeas>::reverse_iterator end_imu_it;
  if (this->imuMeasFromTimeRange(start_time, sorted_timestamps.back(), begin_imu_it, end_imu_it) == false) {
    // not enough IMU measurements, return empty vector
    return empty;
  }

  // Backwards integration to find pose at first IMU sample
  const ImuMeas& f1 = *begin_imu_it;
  const ImuMeas& f2 = *(begin_imu_it+1);

  // Time between first two IMU samples
  double dt = f2.dt;

  // Time between first IMU sample and start_time
  double idt = start_time - f1.stamp;

  // Angular acceleration between first two IMU samples
  Eigen::Vector3f alpha_dt = f2.ang_vel - f1.ang_vel;
  Eigen::Vector3f alpha = alpha_dt / dt;

  // Average angular velocity (reversed) between first IMU sample and start_time
  Eigen::Vector3f omega_i = -(f1.ang_vel + 0.5*alpha*idt);

  // Set q_init to orientation at first IMU sample
  q_init = Eigen::Quaternionf (
    q_init.w() - 0.5*( q_init.x()*omega_i[0] + q_init.y()*omega_i[1] + q_init.z()*omega_i[2] ) * idt,
    q_init.x() + 0.5*( q_init.w()*omega_i[0] - q_init.z()*omega_i[1] + q_init.y()*omega_i[2] ) * idt,
    q_init.y() + 0.5*( q_init.z()*omega_i[0] + q_init.w()*omega_i[1] - q_init.x()*omega_i[2] ) * idt,
    q_init.z() + 0.5*( q_init.x()*omega_i[1] - q_init.y()*omega_i[0] + q_init.w()*omega_i[2] ) * idt
  );
  q_init.normalize();

  // Average angular velocity between first two IMU samples
  Eigen::Vector3f omega = f1.ang_vel + 0.5*alpha_dt;

  // Orientation at second IMU sample
  Eigen::Quaternionf q2 (
    q_init.w() - 0.5*( q_init.x()*omega[0] + q_init.y()*omega[1] + q_init.z()*omega[2] ) * dt,
    q_init.x() + 0.5*( q_init.w()*omega[0] - q_init.z()*omega[1] + q_init.y()*omega[2] ) * dt,
    q_init.y() + 0.5*( q_init.z()*omega[0] + q_init.w()*omega[1] - q_init.x()*omega[2] ) * dt,
    q_init.z() + 0.5*( q_init.x()*omega[1] - q_init.y()*omega[0] + q_init.w()*omega[2] ) * dt
  );
  q2.normalize();

  // Acceleration at first IMU sample
  Eigen::Vector3f a1 = q_init._transformVector(f1.lin_accel);
  a1[2] -= this->gravity_;

  // Acceleration at second IMU sample
  Eigen::Vector3f a2 = q2._transformVector(f2.lin_accel);
  a2[2] -= this->gravity_;

  // Jerk between first two IMU samples
  Eigen::Vector3f j = (a2 - a1) / dt;

  // Set v_init to velocity at first IMU sample (go backwards from start_time)
  v_init -= a1*idt + 0.5*j*idt*idt;

  // Set p_init to position at first IMU sample (go backwards from start_time)
  p_init -= v_init*idt + 0.5*a1*idt*idt + (1/6.)*j*idt*idt*idt;

  return this->integrateImuInternal(q_init, p_init, v_init, sorted_timestamps, begin_imu_it, end_imu_it);
}

std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>>
liom::OdomNode::integrateImuInternal(Eigen::Quaternionf q_init, Eigen::Vector3f p_init, Eigen::Vector3f v_init,
                                     const std::vector<double>& sorted_timestamps,
                                     boost::circular_buffer<ImuMeas>::reverse_iterator begin_imu_it,
                                     boost::circular_buffer<ImuMeas>::reverse_iterator end_imu_it) {

  std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>> imu_se3;

  // Initialization
  Eigen::Quaternionf q = q_init;
  Eigen::Vector3f p = p_init;
  Eigen::Vector3f v = v_init;
  Eigen::Vector3f a = q._transformVector(begin_imu_it->lin_accel);
  a[2] -= this->gravity_;

  // Iterate over IMU measurements and timestamps
  auto prev_imu_it = begin_imu_it;
  auto imu_it = prev_imu_it + 1;

  auto stamp_it = sorted_timestamps.begin();

  for (; imu_it != end_imu_it; imu_it++) {

    const ImuMeas& f0 = *prev_imu_it;
    const ImuMeas& f = *imu_it;

    // Time between IMU samples
    double dt = f.dt;

    // Angular acceleration
    Eigen::Vector3f alpha_dt = f.ang_vel - f0.ang_vel;
    Eigen::Vector3f alpha = alpha_dt / dt;

    // Average angular velocity
    Eigen::Vector3f omega = f0.ang_vel + 0.5*alpha_dt;

    // Orientation
    q = Eigen::Quaternionf (
      q.w() - 0.5*( q.x()*omega[0] + q.y()*omega[1] + q.z()*omega[2] ) * dt,
      q.x() + 0.5*( q.w()*omega[0] - q.z()*omega[1] + q.y()*omega[2] ) * dt,
      q.y() + 0.5*( q.z()*omega[0] + q.w()*omega[1] - q.x()*omega[2] ) * dt,
      q.z() + 0.5*( q.x()*omega[1] - q.y()*omega[0] + q.w()*omega[2] ) * dt
    );
    q.normalize();

    // Acceleration
    Eigen::Vector3f a0 = a;
    a = q._transformVector(f.lin_accel);
    a[2] -= this->gravity_;

    // Jerk
    Eigen::Vector3f j_dt = a - a0;
    Eigen::Vector3f j = j_dt / dt;

    // Interpolate for given timestamps
    while (stamp_it != sorted_timestamps.end() && *stamp_it <= f.stamp) {
      // Time between previous IMU sample and given timestamp
      double idt = *stamp_it - f0.stamp;

      // Average angular velocity
      Eigen::Vector3f omega_i = f0.ang_vel + 0.5*alpha*idt;

      // Orientation
      Eigen::Quaternionf q_i (
        q.w() - 0.5*( q.x()*omega_i[0] + q.y()*omega_i[1] + q.z()*omega_i[2] ) * idt,
        q.x() + 0.5*( q.w()*omega_i[0] - q.z()*omega_i[1] + q.y()*omega_i[2] ) * idt,
        q.y() + 0.5*( q.z()*omega_i[0] + q.w()*omega_i[1] - q.x()*omega_i[2] ) * idt,
        q.z() + 0.5*( q.x()*omega_i[1] - q.y()*omega_i[0] + q.w()*omega_i[2] ) * idt
      );
      q_i.normalize();

      // Position
      Eigen::Vector3f p_i = p + v*idt + 0.5*a0*idt*idt + (1/6.)*j*idt*idt*idt;

      // Transformation
      Eigen::Matrix4f T = Eigen::Matrix4f::Identity();
      T.block(0, 0, 3, 3) = q_i.toRotationMatrix();
      T.block(0, 3, 3, 1) = p_i;

      imu_se3.push_back(T);

      stamp_it++;
    }

    // Position
    p += v*dt + 0.5*a0*dt*dt + (1/6.)*j_dt*dt*dt;
    // Velocity
    v += a0*dt + 0.5*j_dt*dt;
    prev_imu_it = imu_it;
  }

  return imu_se3;

}

void liom::OdomNode::propagateGICP() {

  this->lidarPose.p << this->T(0,3), this->T(1,3), this->T(2,3);

  Eigen::Matrix3f rotSO3;
  rotSO3 << this->T(0,0), this->T(0,1), this->T(0,2),
            this->T(1,0), this->T(1,1), this->T(1,2),
            this->T(2,0), this->T(2,1), this->T(2,2);

  Eigen::Quaternionf q(rotSO3);

  // Normalize quaternion
  double norm = sqrt(q.w()*q.w() + q.x()*q.x() + q.y()*q.y() + q.z()*q.z());
  q.w() /= norm; q.x() /= norm; q.y() /= norm; q.z() /= norm;
  this->lidarPose.q = q;

}

void liom::OdomNode::propagateState() {

  // Lock thread to prevent state from being accessed by UpdateState
  std::lock_guard<std::mutex> lock( this->geo.mtx );

  double dt = this->imu_meas.dt;

  Eigen::Quaternionf qhat = this->state.q, omega;
  Eigen::Vector3f world_accel;

  // Transform accel from body to world frame
  world_accel = qhat._transformVector(this->imu_meas.lin_accel);

  // Accel propogation
  this->state.p[0] += this->state.v.lin.w[0]*dt + 0.5*dt*dt*world_accel[0];
  this->state.p[1] += this->state.v.lin.w[1]*dt + 0.5*dt*dt*world_accel[1];
  this->state.p[2] += this->state.v.lin.w[2]*dt + 0.5*dt*dt*(world_accel[2] - this->gravity_);

  this->state.v.lin.w[0] += world_accel[0]*dt;
  this->state.v.lin.w[1] += world_accel[1]*dt;
  this->state.v.lin.w[2] += (world_accel[2] - this->gravity_)*dt;
  this->state.v.lin.b = this->state.q.toRotationMatrix().inverse() * this->state.v.lin.w;

  // Gyro propogation
  omega.w() = 0;
  omega.vec() = this->imu_meas.ang_vel;
  Eigen::Quaternionf tmp = qhat * omega;
  this->state.q.w() += 0.5 * dt * tmp.w();
  this->state.q.vec() += 0.5 * dt * tmp.vec();

  // Ensure quaternion is properly normalized
  this->state.q.normalize();

  this->state.v.ang.b = this->imu_meas.ang_vel;
  this->state.v.ang.w = this->state.q.toRotationMatrix() * this->state.v.ang.b;
}

void liom::OdomNode::updateState() {

  // Lock thread to prevent state from being accessed by PropagateState
  std::lock_guard<std::mutex> lock( this->geo.mtx );

  Eigen::Vector3f pin = this->lidarPose.p;
  Eigen::Quaternionf qin = this->lidarPose.q;
  double dt = this->scan_stamp - this->prev_scan_stamp;

  /*
    Modification
  */
  Eigen::Matrix<float, 12, 12> F_cv, F_ca, F_ct;
  F_cv = Eigen::Matrix<float, 12, 12>::Identity(); 
  F_ca = Eigen::Matrix<float, 12, 12>::Identity(); 
  F_ct = Eigen::Matrix<float, 12, 12>::Identity(); 

  Eigen::Vector3f angular_vel = this->angl_total / 3.14 * 180 / this->num_imu;
  angular_vel = angular_vel / this->gravity_;
  Eigen::Vector3f accel_avg   = this->accel_total / this->num_imu;
  accel_avg[2] -= this->gravity_;
  float lidar_dt = this->scan_stamp - this->prev_scan_stamp;
  // Compute transformation
  F_cv.block<3,3>(0,3) = Eigen::Matrix<float, 3,3>::Identity() * lidar_dt;

  // CA model
  F_ca.block<3,3>(0,3) = Eigen::Matrix<float, 3,3>::Identity() * lidar_dt; 
  F_ca.block<3,3>(3,9) = Eigen::Matrix<float, 3,3>::Identity() * lidar_dt; 
  F_ca.block<3,3>(0,9) = Eigen::Matrix<float, 3,3>::Identity() * 0.5 * lidar_dt * lidar_dt; 

  // CT model
  float Tx = angular_vel[0];
  float Ty = angular_vel[1];
  float Tz = angular_vel[2];
  float T_xyz = sqrt(Tx*Tx + Ty*Ty + Tz*Tz);

  float c1 = (cos(T_xyz * lidar_dt) - 1)/(T_xyz*T_xyz);
  float c2    = sin(T_xyz*lidar_dt)/T_xyz;
  float c3    = (1/(T_xyz*T_xyz))*(sin(T_xyz)*lidar_dt)/(T_xyz - lidar_dt);
  float d1    = Ty*Ty + Tz*Tz;
  float d2    = Tx*Tx + Tz*Tz;
  float d3    = Tx*Tx + Ty*Ty;
  Eigen::Matrix<float, 3, 3> A;
  Eigen::Matrix<float, 3, 3> B;
  A = Eigen::Matrix<float, 3, 3>::Zero();
  B = Eigen::Matrix<float, 3, 3>::Zero();

  A.block<1,3>(0,0) << c1*d1, -c2*Tz-c1*Tx*Ty, c2*Ty-c1*Tx*Tz;
  A.block<1,3>(1,0) << c2*Tz-c1*Tx*Ty, c1*d2, -c2*Tx-c1*Ty*Tz;
  A.block<1,3>(2,0) << -c2*Ty-c1*Tx*Tz, c2*Tx-c1*Ty*Tz, c1*d3;

  B.block<1,3>(0,0) << c3*d1, c1*Tz-c3*Tx*Ty, -c1*Ty-c3*Tx*Tz;
  B.block<1,3>(1,0) << -c1*Tz-c3*Tx*Ty, c3*d2, c1*Tx-c3*Ty*Tz;
  B.block<1,3>(2,0) << c1*Ty-c3*Tx*Tz, -c1*Tx-c3*Ty*Tz, c3*d3;

  F_ct.block<3,3>(0,3) = B;
  F_ct.block<3,3>(3,3) += A;

  // Predict state for each model
  Eigen::Matrix<float, 12, 1> x_pred_cv, x_pred_ca, x_pred_ct, x_hat_cv, x_hat_ca, x_hat_ct, x_imm, x_imm_est;
  if (!std::isnan(angular_vel.x()) && !std::isnan(angular_vel.y()) && !std::isnan(angular_vel.z()) && 
    !std::isnan(accel_avg.x())   && !std::isnan(accel_avg.y())   && !std::isnan(accel_avg.z())) {
    x_imm << this->state.p, this->state.v.lin.w, angular_vel, accel_avg;
  
    x_pred_cv = F_cv * x_imm;
    x_pred_ca = F_ca * x_imm;
    x_pred_ct = F_ct * x_imm;
    x_pred_ct(2) = x_pred_cv(2);
    Eigen::Matrix<float, 3, 12> H;
    H  = Eigen::Matrix<float, 3, 12>::Zero();
    H.block<3,3>(0,0) = Eigen::Matrix<float, 3, 3>::Identity();    
    Eigen::Matrix<float, 12, 12> P_pred_cv, P_pred_ca, P_pred_ct, P_imm, Q;
    P_imm = Eigen::Matrix<float, 12, 12>::Identity()*0.00001;
    Q     = Eigen::Matrix<float, 12, 12>::Identity()*0.00001;

    // Predict covariance matrix
    P_pred_cv = F_cv * P_imm * F_cv.transpose() + Q;
    P_pred_ca = F_ca * P_imm * F_ca.transpose() + Q;
    P_pred_ct = F_ct * P_imm * F_ct.transpose() + Q;

    // Compute residual covariance
    Eigen::Matrix<float, 3, 3> S_cv, S_ca, S_ct, P_ij;

    S_cv = H * P_pred_cv * H.transpose() + this->R;
    S_ca = H * P_pred_ca * H.transpose() + this->R;
    S_ct = H * P_pred_ct * H.transpose() + this->R;

    Eigen::Matrix<float, 12, 3> K_cv, K_ca, K_ct;
    K_cv = P_pred_cv * H.transpose() * (H * P_pred_cv * H.transpose() + this->R).inverse();
    K_ca = P_pred_ca * H.transpose() * (H * P_pred_ca * H.transpose() + this->R).inverse();
    K_ct = P_pred_ct * H.transpose() * (H * P_pred_ct * H.transpose() + this->R).inverse();

    // Compute residual error to LiDAR measurments
    Eigen::Matrix<float, 3, 1> err_cv, err_ca, err_ct, cbar, Mu_ij;
    P_ij  << 0.9, 0.01, 0.09, 0.025, 0.75, 0.225, 0.075, 0.175, 0.75;
    Mu_ij << 0.333, 0.333, 0.333;
    cbar   = P_ij * Mu_ij;
    err_cv = pin - H*x_pred_cv;
    err_ca = pin - H*x_pred_ca;
    err_ct = pin - H*x_pred_ct;
    // Compute the likelihood score of each model compared to meas under Gaussian
    float pi = 3.14159265359;
    float Lfun_cv, Lfun_ca, Lfun_ct;
    // Compute log-likelihoods (stable)      
    Lfun_cv = 1/sqrt(abs(2*pi * S_cv.determinant())) * exp((-0.5 * err_cv.transpose() * S_cv.inverse() * err_cv));
    Lfun_ca = 1/sqrt(abs(2*pi * S_ca.determinant())) * exp((-0.5 * err_ca.transpose() * S_ca.inverse() * err_ca));
    Lfun_ct = 1/sqrt(abs(2*pi * S_ct.determinant())) * exp((-0.5 * err_ct.transpose() * S_ct.inverse() * err_ct));

    // Update
    x_hat_cv = x_pred_cv + K_cv * err_cv;
    x_hat_ca = x_pred_ca + K_ca * err_ca;
    x_hat_ct = x_pred_ct + K_ct * err_ct;
    x_hat_ct(2) = x_hat_cv(2);

    if (Lfun_cv == 0 && Lfun_ca == 0 && Lfun_ct == 0) {
      std::cout << "Debug" << std::endl;
    } else {
      // Mixing params probability
      float t_Lfun = Lfun_cv + Lfun_ca + Lfun_ct + 1e-10f;
      float Lfun_cv_norm = Lfun_cv / t_Lfun;
      float Lfun_ca_norm = Lfun_ca / t_Lfun;
      float Lfun_ct_norm = Lfun_ct / t_Lfun;

      Eigen::Matrix<float, 1, 3> merge;
      merge << Lfun_cv_norm, Lfun_ca_norm, Lfun_ct_norm;
      float c = merge * cbar;
      Mu_ij   = merge.transpose().cwiseProduct(cbar/c);
    }

    x_imm_est = x_hat_cv * Mu_ij[0] + x_hat_ca * Mu_ij[1] + x_hat_ct * Mu_ij[2];
  }
  // Reset params
  this->accel_total << 0, 0, 0;
  this->angl_total  << 0, 0, 0;
  this->num_imu      = 0;


  Eigen::Quaternionf qe, qhat, qcorr;
  qhat = this->state.q;

  // Constuct error quaternion
  qe = qhat.conjugate()*qin;

  double sgn = 1.;
  if (qe.w() < 0) {
    sgn = -1;
  }

  // Construct quaternion correction
  qcorr.w() = 1 - abs(qe.w());
  qcorr.vec() = sgn*qe.vec();
  qcorr = qhat * qcorr;

  Eigen::Vector3f err = pin - this->state.p;
  Eigen::Vector3f err_body;


  err_body = qhat.conjugate()._transformVector(err);

  double abias_max = this->geo_abias_max_;
  double gbias_max = this->geo_gbias_max_;

  // Update accel bias
  this->state.b.accel -= dt * this->geo_Kab_ * err_body;
  this->state.b.accel = this->state.b.accel.array().min(abias_max).max(-abias_max);

  // Update gyro bias
  this->state.b.gyro[0] -= dt * this->geo_Kgb_ * qe.w() * qe.x();
  this->state.b.gyro[1] -= dt * this->geo_Kgb_ * qe.w() * qe.y();
  this->state.b.gyro[2] -= dt * this->geo_Kgb_ * qe.w() * qe.z();
  this->state.b.gyro = this->state.b.gyro.array().min(gbias_max).max(-gbias_max);

  // Update state
  this->state.p += dt * this->geo_Kp_ * err;
  this->state.v.lin.w += dt * this->geo_Kv_ * err;
  this->state.q.w() += dt * this->geo_Kq_ * qcorr.w();
  this->state.q.x() += dt * this->geo_Kq_ * qcorr.x();
  this->state.q.y() += dt * this->geo_Kq_ * qcorr.y();
  this->state.q.z() += dt * this->geo_Kq_ * qcorr.z();
  this->state.q.normalize();

  // store previous pose, orientation, and velocity
  this->geo.prev_p = this->state.p;
  this->geo.prev_q = this->state.q;
  this->geo.prev_vel = this->state.v.lin.w;

}

sensor_msgs::msg::Imu::SharedPtr liom::OdomNode::transformImu(const sensor_msgs::msg::Imu::SharedPtr& imu_raw) {

  auto imu = std::make_shared<sensor_msgs::msg::Imu>();

  // Copy header
  imu->header = imu_raw->header;

  double imu_stamp_secs = rclcpp::Time(imu->header.stamp).seconds();
  static double prev_stamp = imu_stamp_secs;
  double dt = imu_stamp_secs - prev_stamp;
  prev_stamp = imu_stamp_secs;
  
  if (dt == 0) { dt = 1.0/200.0; }

  // Transform angular velocity (will be the same on a rigid body, so just rotate to ROS convention)
  Eigen::Vector3f ang_vel(imu_raw->angular_velocity.x,
                          imu_raw->angular_velocity.y,
                          imu_raw->angular_velocity.z);

  Eigen::Vector3f ang_vel_cg = this->extrinsics.baselink2imu.R * ang_vel;

  imu->angular_velocity.x = ang_vel_cg[0];
  imu->angular_velocity.y = ang_vel_cg[1];
  imu->angular_velocity.z = ang_vel_cg[2];

  static Eigen::Vector3f ang_vel_cg_prev = ang_vel_cg;

  // Transform linear acceleration (need to account for component due to translational difference)
  Eigen::Vector3f lin_accel(imu_raw->linear_acceleration.x,
                            imu_raw->linear_acceleration.y,
                            imu_raw->linear_acceleration.z);

  Eigen::Vector3f lin_accel_cg = this->extrinsics.baselink2imu.R * lin_accel;

  lin_accel_cg = lin_accel_cg
                 + ((ang_vel_cg - ang_vel_cg_prev) / dt).cross(-this->extrinsics.baselink2imu.t)
                 + ang_vel_cg.cross(ang_vel_cg.cross(-this->extrinsics.baselink2imu.t));

  ang_vel_cg_prev = ang_vel_cg;

  imu->linear_acceleration.x = lin_accel_cg[0];
  imu->linear_acceleration.y = lin_accel_cg[1];
  imu->linear_acceleration.z = lin_accel_cg[2];

  return imu;

}

void liom::OdomNode::computeMetrics() {
  this->computeSpaciousness();
  this->computeDensity();
}

void liom::OdomNode::computeSpaciousness() {

  // compute range of points
  std::vector<float> ds;

  for (int i = 0; i <= this->original_scan->points.size(); i++) {
    float d = std::sqrt(pow(this->original_scan->points[i].x, 2) +
                        pow(this->original_scan->points[i].y, 2));
    ds.push_back(d);
  }

  // median
  std::nth_element(ds.begin(), ds.begin() + ds.size()/2, ds.end());
  float median_curr = ds[ds.size()/2];
  static float median_prev = median_curr;
  float median_lpf = 0.95*median_prev + 0.05*median_curr;
  median_prev = median_lpf;

  // push
  this->metrics.spaciousness.push_back( median_lpf );

}

void liom::OdomNode::computeDensity() {

  float density;

  if (!this->geo.first_opt_done) {
    density = 0.;
  } else {
    density = this->gicp.source_density_;
  }

  static float density_prev = density;
  float density_lpf = 0.95*density_prev + 0.05*density;
  density_prev = density_lpf;

  this->metrics.density.push_back( density_lpf );

}

void liom::OdomNode::computeConvexHull() {

  // at least 4 keyframes for convex hull
  if (this->num_processed_keyframes < 4) {
    return;
  }

  // create a pointcloud with points at keyframes
  pcl::PointCloud<PointType>::Ptr cloud = std::make_shared<pcl::PointCloud<PointType>>();

  std::unique_lock<decltype(this->keyframes_mutex)> lock(this->keyframes_mutex);
  for (int i = 0; i < this->num_processed_keyframes; i++) {
    PointType pt;
    pt.x = this->keyframes[i].first.first[0];
    pt.y = this->keyframes[i].first.first[1];
    pt.z = this->keyframes[i].first.first[2];
    cloud->push_back(pt);
  }
  lock.unlock();

  // calculate the convex hull of the point cloud
  this->convex_hull.setInputCloud(cloud);

  // get the indices of the keyframes on the convex hull
  pcl::PointCloud<PointType>::Ptr convex_points = std::make_shared<pcl::PointCloud<PointType>>();
  this->convex_hull.reconstruct(*convex_points);

  pcl::PointIndices::Ptr convex_hull_point_idx = std::make_shared<pcl::PointIndices>();
  this->convex_hull.getHullPointIndices(*convex_hull_point_idx);

  this->keyframe_convex.clear();
  for (int i=0; i<convex_hull_point_idx->indices.size(); ++i) {
    this->keyframe_convex.push_back(convex_hull_point_idx->indices[i]);
  }

}

void liom::OdomNode::computeConcaveHull() {

  // at least 5 keyframes for concave hull
  if (this->num_processed_keyframes < 5) {
    return;
  }

  // create a pointcloud with points at keyframes
  auto cloud = std::make_shared<pcl::PointCloud<PointType>>();

  std::unique_lock<decltype(this->keyframes_mutex)> lock(this->keyframes_mutex);
  for (int i = 0; i < this->num_processed_keyframes; i++) {
    PointType pt;
    pt.x = this->keyframes[i].first.first[0];
    pt.y = this->keyframes[i].first.first[1];
    pt.z = this->keyframes[i].first.first[2];
    cloud->push_back(pt);
  }
  lock.unlock();

  // calculate the concave hull of the point cloud
  this->concave_hull.setInputCloud(cloud);

  // get the indices of the keyframes on the concave hull
  pcl::PointCloud<PointType>::Ptr concave_points = std::make_shared<pcl::PointCloud<PointType>>();
  this->concave_hull.reconstruct(*concave_points);

  pcl::PointIndices::Ptr concave_hull_point_idx = std::make_shared<pcl::PointIndices>();
  this->concave_hull.getHullPointIndices(*concave_hull_point_idx);

  this->keyframe_concave.clear();
  for (int i=0; i<concave_hull_point_idx->indices.size(); ++i) {
    this->keyframe_concave.push_back(concave_hull_point_idx->indices[i]);
  }

}

void liom::OdomNode::updateKeyframes() {

  // calculate difference in pose and rotation to all poses in trajectory
  float closest_d = std::numeric_limits<float>::infinity();
  int closest_idx = 0;
  int keyframes_idx = 0;

  int num_nearby = 0;

  for (const auto& k : this->keyframes) {

    // calculate distance between current pose and pose in keyframes
    float delta_d = sqrt( pow(this->state.p[0] - k.first.first[0], 2) +
                          pow(this->state.p[1] - k.first.first[1], 2) +
                          pow(this->state.p[2] - k.first.first[2], 2) );

    // count the number nearby current pose
    if (delta_d <= this->keyframe_thresh_dist_ * 1.5){
      ++num_nearby;
    }

    // store into variable
    if (delta_d < closest_d) {
      closest_d = delta_d;
      closest_idx = keyframes_idx;
    }

    keyframes_idx++;
    this->mahalanobis_idx_ = true;
  }

  // get closest pose and corresponding rotation
  Eigen::Vector3f closest_pose = this->keyframes[closest_idx].first.first;
  Eigen::Quaternionf closest_pose_r = this->keyframes[closest_idx].first.second;

  // calculate distance between current pose and closest pose from above
  float dd = sqrt( pow(this->state.p[0] - closest_pose[0], 2) +
                   pow(this->state.p[1] - closest_pose[1], 2) +
                   pow(this->state.p[2] - closest_pose[2], 2) );

  // calculate difference in orientation using SLERP
  Eigen::Quaternionf dq;

  if (this->state.q.dot(closest_pose_r) < 0.) {
    Eigen::Quaternionf lq = closest_pose_r;
    lq.w() *= -1.; lq.x() *= -1.; lq.y() *= -1.; lq.z() *= -1.;
    dq = this->state.q * lq.inverse();
  } else {
    dq = this->state.q * closest_pose_r.inverse();
  }

  double theta_rad = 2. * atan2(sqrt( pow(dq.x(), 2) + pow(dq.y(), 2) + pow(dq.z(), 2) ), dq.w());
  double theta_deg = theta_rad * (180.0/M_PI);

  // update keyframes
  bool newKeyframe = false;

  if (abs(dd) > this->keyframe_thresh_dist_ || abs(theta_deg) > this->keyframe_thresh_rot_) {
    newKeyframe = true;
  }

  if (abs(dd) <= this->keyframe_thresh_dist_) {
    newKeyframe = false;
  }

  if (abs(dd) <= this->keyframe_thresh_dist_ && abs(theta_deg) > this->keyframe_thresh_rot_ && num_nearby <= 1) {
    newKeyframe = true;
  }

  if (newKeyframe) {

    // update keyframe vector
    std::unique_lock<decltype(this->keyframes_mutex)> lock(this->keyframes_mutex);
    this->keyframes.push_back(std::make_pair(std::make_pair(this->lidarPose.p, this->lidarPose.q), this->current_scan));
    this->keyframe_timestamps.push_back(this->scan_header_stamp);
    this->keyframe_normals.push_back(this->gicp.getSourceCovariances());
    this->keyframe_transformations.push_back(this->T_corr);
    lock.unlock();

  }

}

void liom::OdomNode::setAdaptiveParams() {

  // Spaciousness
  float sp = this->metrics.spaciousness.back();

  if (sp < 0.5) { sp = 0.5; }
  if (sp > 5.0) { sp = 5.0; }

  this->keyframe_thresh_dist_ = sp;

  // Density
  float den = this->metrics.density.back();

  if (den < 0.5*this->gicp_max_corr_dist_) { den = 0.5*this->gicp_max_corr_dist_; }
  if (den > 2.0*this->gicp_max_corr_dist_) { den = 2.0*this->gicp_max_corr_dist_; }

  if (sp < 5.0) { den = 0.5*this->gicp_max_corr_dist_; };
  if (sp > 5.0) { den = 2.0*this->gicp_max_corr_dist_; };

  this->gicp.setMaxCorrespondenceDistance(den);

  // Concave hull alpha
  this->concave_hull.setAlpha(this->keyframe_thresh_dist_);

}

void liom::OdomNode::pushSubmapIndices(std::vector<float> dists, int k, std::vector<int> frames) {

  // make sure dists is not empty
  if (!dists.size()) { return; }

  // maintain max heap of at most k elements
  std::priority_queue<float> pq;

  for (auto d : dists) {
    if (pq.size() >= k && pq.top() > d) {
      pq.push(d);
      pq.pop();
    } else if (pq.size() < k) {
      pq.push(d);
    }
  }

  // get the kth smallest element, which should be at the top of the heap
  float kth_element = pq.top();

  // get all elements smaller or equal to the kth smallest element
  for (int i = 0; i < dists.size(); ++i) {
    if (dists[i] <= kth_element)
      this->submap_kf_idx_curr.push_back(frames[i]);
  }

}

void liom::OdomNode::buildSubmap(State vehicle_state) {

  // clear vector of keyframe indices to use for submap
  this->submap_kf_idx_curr.clear();

  // calculate distance between current pose and poses in keyframe set
  std::unique_lock<decltype(this->keyframes_mutex)> lock(this->keyframes_mutex);
  std::vector<float> ds;
  std::vector<int> keyframe_nn;
  for (int i = 0; i < this->num_processed_keyframes; i++) {
    float d = sqrt( pow(vehicle_state.p[0] - this->keyframes[i].first.first[0], 2) +
                    pow(vehicle_state.p[1] - this->keyframes[i].first.first[1], 2) +
                    pow(vehicle_state.p[2] - this->keyframes[i].first.first[2], 2) );
    ds.push_back(d);
    keyframe_nn.push_back(i);
  }
  lock.unlock();

  // get indices for top K nearest neighbor keyframe poses
  this->pushSubmapIndices(ds, this->submap_knn_, keyframe_nn);

  // get convex hull indices
  this->computeConvexHull();

  // get distances for each keyframe on convex hull
  std::vector<float> convex_ds;
  for (const auto& c : this->keyframe_convex) {
    convex_ds.push_back(ds[c]);
  }

  // get indices for top kNN for convex hull
  this->pushSubmapIndices(convex_ds, this->submap_kcv_, this->keyframe_convex);

  // get concave hull indices
  this->computeConcaveHull();

  // get distances for each keyframe on concave hull
  std::vector<float> concave_ds;
  for (const auto& c : this->keyframe_concave) {
    concave_ds.push_back(ds[c]);
  }

  // get indices for top kNN for concave hull
  this->pushSubmapIndices(concave_ds, this->submap_kcc_, this->keyframe_concave);

  // sort current and previous submap kf list of indices
  std::sort(this->submap_kf_idx_curr.begin(), this->submap_kf_idx_curr.end());
  std::sort(this->submap_kf_idx_prev.begin(), this->submap_kf_idx_prev.end());

  // remove duplicate indices
  auto last = std::unique(this->submap_kf_idx_curr.begin(), this->submap_kf_idx_curr.end());
  this->submap_kf_idx_curr.erase(last, this->submap_kf_idx_curr.end());

  // check if submap has changed from previous iteration
  if (this->submap_kf_idx_curr != this->submap_kf_idx_prev){

    this->submap_hasChanged = true;

    // Pause to prevent stealing resources from the main loop if it is running.
    this->pauseSubmapBuildIfNeeded();

    // reinitialize submap cloud and normals
    pcl::PointCloud<PointType>::Ptr submap_cloud_ = std::make_shared<pcl::PointCloud<PointType>>();
    std::shared_ptr<nano_gicp::CovarianceList> submap_normals_ (std::make_shared<nano_gicp::CovarianceList>());

    for (auto k : this->submap_kf_idx_curr) {

      // create current submap cloud
      lock.lock();
      *submap_cloud_ += *this->keyframes[k].second;
      lock.unlock();

      // grab corresponding submap cloud's normals
      submap_normals_->insert( std::end(*submap_normals_),
          std::begin(*(this->keyframe_normals[k])), std::end(*(this->keyframe_normals[k])) );
    }

    this->submap_cloud = submap_cloud_;
    this->submap_normals = submap_normals_;

    // Modification
    Eigen::Matrix4d mat_curr = this->submap_normals->back();
    Eigen::Matrix3d mat_cov  = mat_curr.block<3,3>(0,0);

    // Convert to float
    Eigen::Matrix3f mat_cov_fl = mat_cov.cast<float>();
    this->R = mat_cov_fl;

    // Pause to prevent stealing resources from the main loop if it is running.
    this->pauseSubmapBuildIfNeeded();

    this->gicp_temp.setInputTarget(this->submap_cloud);
    this->submap_kdtree = this->gicp_temp.target_kdtree_;

    this->submap_kf_idx_prev = this->submap_kf_idx_curr;
  }
}

void liom::OdomNode::buildKeyframesAndSubmap(State vehicle_state) {

  // transform the new keyframe(s) and associated covariance list(s)
    std::unique_lock<decltype(this->keyframes_mutex)> lock(this->keyframes_mutex);

  for (int i = this->num_processed_keyframes; i < this->keyframes.size(); i++) {
    pcl::PointCloud<PointType>::ConstPtr raw_keyframe = this->keyframes[i].second;
    std::shared_ptr<const nano_gicp::CovarianceList> raw_covariances = this->keyframe_normals[i];
    Eigen::Matrix4f T = this->keyframe_transformations[i];
    lock.unlock();

    Eigen::Matrix4d Td = T.cast<double>();

    pcl::PointCloud<PointType>::Ptr transformed_keyframe = std::make_shared<pcl::PointCloud<PointType>>();
    pcl::transformPointCloud (*raw_keyframe, *transformed_keyframe, T);

    std::shared_ptr<nano_gicp::CovarianceList> transformed_covariances (std::make_shared<nano_gicp::CovarianceList>(raw_covariances->size()));
    std::transform(raw_covariances->begin(), raw_covariances->end(), transformed_covariances->begin(),
                   [&Td](Eigen::Matrix4d cov) { return Td * cov * Td.transpose(); });

    ++this->num_processed_keyframes;

    lock.lock();
    this->keyframes[i].second = transformed_keyframe;
    this->keyframe_normals[i] = transformed_covariances;

    this->publish_keyframe_thread = std::thread( &liom::OdomNode::publishKeyframe, this, this->keyframes[i], this->keyframe_timestamps[i] );
    this->publish_keyframe_thread.detach();
  }

  lock.unlock();

  // Pause to prevent stealing resources from the main loop if it is running.
  this->pauseSubmapBuildIfNeeded();

  this->buildSubmap(vehicle_state);
}

void liom::OdomNode::pauseSubmapBuildIfNeeded() {
  std::unique_lock<decltype(this->main_loop_running_mutex)> lock(this->main_loop_running_mutex);
  this->submap_build_cv.wait(lock, [this]{ return !this->main_loop_running; });
}

void liom::OdomNode::debug() {

  // Total length traversed
  double length_traversed = 0.;
  Eigen::Vector3f p_curr = Eigen::Vector3f(0., 0., 0.);
  Eigen::Vector3f p_prev = Eigen::Vector3f(0., 0., 0.);
  for (const auto& t : this->trajectory) {
    if (p_prev == Eigen::Vector3f(0., 0., 0.)) {
      p_prev = t.first;
      continue;
    }
    p_curr = t.first;
    double l = sqrt(pow(p_curr[0] - p_prev[0], 2) + pow(p_curr[1] - p_prev[1], 2) + pow(p_curr[2] - p_prev[2], 2));

    if (l >= 0.1) {
      length_traversed += l;
      p_prev = p_curr;
    }
  }
  this->length_traversed = length_traversed;

  // Average computation time
  double avg_comp_time =
    std::accumulate(this->comp_times.begin(), this->comp_times.end(), 0.0) / this->comp_times.size();

  // Average sensor rates
  int win_size = 100;
  double avg_imu_rate;
  double avg_lidar_rate;
  if (this->imu_rates.size() < win_size) {
    avg_imu_rate =
      std::accumulate(this->imu_rates.begin(), this->imu_rates.end(), 0.0) / this->imu_rates.size();
  } else {
    avg_imu_rate =
      std::accumulate(this->imu_rates.end()-win_size, this->imu_rates.end(), 0.0) / win_size;
  }
  if (this->lidar_rates.size() < win_size) {
    avg_lidar_rate =
      std::accumulate(this->lidar_rates.begin(), this->lidar_rates.end(), 0.0) / this->lidar_rates.size();
  } else {
    avg_lidar_rate =
      std::accumulate(this->lidar_rates.end()-win_size, this->lidar_rates.end(), 0.0) / win_size;
  }

  // RAM Usage
  double vm_usage = 0.0;
  double resident_set = 0.0;
  std::ifstream stat_stream("/proc/self/stat", std::ios_base::in); //get info from proc directory
  std::string pid, comm, state, ppid, pgrp, session, tty_nr;
  std::string tpgid, flags, minflt, cminflt, majflt, cmajflt;
  std::string utime, stime, cutime, cstime, priority, nice;
  std::string num_threads, itrealvalue, starttime;
  unsigned long vsize;
  long rss;
  stat_stream >> pid >> comm >> state >> ppid >> pgrp >> session >> tty_nr
              >> tpgid >> flags >> minflt >> cminflt >> majflt >> cmajflt
              >> utime >> stime >> cutime >> cstime >> priority >> nice
              >> num_threads >> itrealvalue >> starttime >> vsize >> rss; // don't care about the rest
  stat_stream.close();
  long page_size_kb = sysconf(_SC_PAGE_SIZE) / 1024; // for x86-64 is configured to use 2MB pages
  vm_usage = vsize / 1024.0;
  resident_set = rss * page_size_kb;

  // CPU Usage
  struct tms timeSample;
  clock_t now;
  double cpu_percent;
  now = times(&timeSample);
  if (now <= this->lastCPU || timeSample.tms_stime < this->lastSysCPU ||
      timeSample.tms_utime < this->lastUserCPU) {
      cpu_percent = -1.0;
  } else {
      cpu_percent = (timeSample.tms_stime - this->lastSysCPU) + (timeSample.tms_utime - this->lastUserCPU);
      cpu_percent /= (now - this->lastCPU);
      cpu_percent /= this->numProcessors;
      cpu_percent *= 100.;
  }
  this->lastCPU = now;
  this->lastSysCPU = timeSample.tms_stime;
  this->lastUserCPU = timeSample.tms_utime;
  this->cpu_percents.push_back(cpu_percent);
  double avg_cpu_usage =
    std::accumulate(this->cpu_percents.begin(), this->cpu_percents.end(), 0.0) / this->cpu_percents.size();

  // Saving odometry in strict TUM format
  std::ofstream outFile("/home/ara/ros2_ws/src/lio_imm/results/trajectory.txt", std::ios::app);
  if (!outFile.is_open()) {
      RCLCPP_ERROR(this->get_logger(), "Failed to open trajectory file!");
      return;
  }

  // Get timestamp in seconds.nanoseconds format
  int64_t total_ns = this->imu_stamp.nanoseconds();
  int64_t sec = total_ns / 1000000000;
  int64_t nsec = total_ns % 1000000000;

  // Write in strict TUM format (8 space-separated values)
  outFile << sec << "." << std::setw(9) << std::setfill('0') << nsec << " " 
          << std::setprecision(9) << this->state.p[0] << " " 
          << this->state.p[1] << " " 
          << this->state.p[2] << " "
          << this->state.q.x() << " "
          << this->state.q.y() << " "
          << this->state.q.z() << " "
          << this->state.q.w() << "\n";  // Use \n instead of endl to avoid unnecessary flush

  outFile.close();

  // Print to terminal
  printf("\033[2J\033[1;1H");

  std::cout << std::endl
            << "+-------------------------------------------------------------------+" << std::endl;
  std::cout << "+-------------------------------------------------------------------+" << std::endl;

  std::time_t curr_time = this->scan_stamp;
  std::string asc_time = std::asctime(std::localtime(&curr_time)); asc_time.pop_back();
  std::cout << "| " << std::left << asc_time;
  std::cout << std::right << std::setfill(' ') << std::setw(42)
    << "Elapsed Time: " + to_string_with_precision(this->elapsed_time, 2) + " seconds "
    << "|" << std::endl;

  if ( !this->cpu_type.empty() ) {
    std::cout << "| " << std::left << std::setfill(' ') << std::setw(66)
      << this->cpu_type + " x " + std::to_string(this->numProcessors)
      << "|" << std::endl;
  }

  if (this->sensor == liom::SensorType::OUSTER) {
    std::cout << "| " << std::left << std::setfill(' ') << std::setw(66)
      << "Sensor Rates: Ouster @ " + to_string_with_precision(avg_lidar_rate, 2)
                                   + " Hz, IMU @ " + to_string_with_precision(avg_imu_rate, 2) + " Hz"
      << "|" << std::endl;
  } else if (this->sensor == liom::SensorType::VELODYNE) {
    std::cout << "| " << std::left << std::setfill(' ') << std::setw(66)
      << "Sensor Rates: Velodyne @ " + to_string_with_precision(avg_lidar_rate, 2)
                                     + " Hz, IMU @ " + to_string_with_precision(avg_imu_rate, 2) + " Hz"
      << "|" << std::endl;
  } else if (this->sensor == liom::SensorType::HESAI) {
    std::cout << "| " << std::left << std::setfill(' ') << std::setw(66)
      << "Sensor Rates: Hesai @ " + to_string_with_precision(avg_lidar_rate, 2)
                                  + " Hz, IMU @ " + to_string_with_precision(avg_imu_rate, 2) + " Hz"
      << "|" << std::endl;
  } else if (this->sensor == liom::SensorType::LIVOX) {
    std::cout << "| " << std::left << std::setfill(' ') << std::setw(66)
      << "Sensor Rates: Livox @ " + to_string_with_precision(avg_lidar_rate, 2)
                                  + " Hz, IMU @ " + to_string_with_precision(avg_imu_rate, 2) + " Hz"
      << "|" << std::endl;
  } else {
    std::cout << "| " << std::left << std::setfill(' ') << std::setw(66)
      << "Sensor Rates: Unknown LiDAR @ " + to_string_with_precision(avg_lidar_rate, 2)
                                          + " Hz, IMU @ " + to_string_with_precision(avg_imu_rate, 2) + " Hz"
      << "|" << std::endl;
  }

  std::cout << "|===================================================================|" << std::endl;

  std::cout << "| " << std::left << std::setfill(' ') << std::setw(66)
    << "Position     {W}  [xyz] :: " + to_string_with_precision(this->state.p[0], 4) + " "
                                + to_string_with_precision(this->state.p[1], 4) + " "
                                + to_string_with_precision(this->state.p[2], 4)
    << "|" << std::endl;
  std::cout << "| " << std::left << std::setfill(' ') << std::setw(66)
    << "Orientation  {W} [wxyz] :: " + to_string_with_precision(this->state.q.w(), 4) + " "
                                + to_string_with_precision(this->state.q.x(), 4) + " "
                                + to_string_with_precision(this->state.q.y(), 4) + " "
                                + to_string_with_precision(this->state.q.z(), 4)
    << "|" << std::endl;
  std::cout << "| " << std::left << std::setfill(' ') << std::setw(66)
    << "Lin Velocity {B}  [xyz] :: " + to_string_with_precision(this->state.v.lin.b[0], 4) + " "
                                + to_string_with_precision(this->state.v.lin.b[1], 4) + " "
                                + to_string_with_precision(this->state.v.lin.b[2], 4)
    << "|" << std::endl;
  std::cout << "| " << std::left << std::setfill(' ') << std::setw(66)
    << "Ang Velocity {B}  [xyz] :: " + to_string_with_precision(this->state.v.ang.b[0], 4) + " "
                                + to_string_with_precision(this->state.v.ang.b[1], 4) + " "
                                + to_string_with_precision(this->state.v.ang.b[2], 4)
    << "|" << std::endl;
  std::cout << "| " << std::left << std::setfill(' ') << std::setw(66)
    << "Accel Bias        [xyz] :: " + to_string_with_precision(this->state.b.accel[0], 8) + " "
                                + to_string_with_precision(this->state.b.accel[1], 8) + " "
                                + to_string_with_precision(this->state.b.accel[2], 8)
    << "|" << std::endl;
  std::cout << "| " << std::left << std::setfill(' ') << std::setw(66)
    << "Gyro Bias         [xyz] :: " + to_string_with_precision(this->state.b.gyro[0], 8) + " "
                                + to_string_with_precision(this->state.b.gyro[1], 8) + " "
                                + to_string_with_precision(this->state.b.gyro[2], 8)
    << "|" << std::endl;

  std::cout << "|                                                                   |" << std::endl;

  std::cout << "| " << std::left << std::setfill(' ') << std::setw(66)
    << "Distance Traveled  :: " + to_string_with_precision(length_traversed, 4) + " meters"
    << "|" << std::endl;
  std::cout << "| " << std::left << std::setfill(' ') << std::setw(66)
    << "Distance to Origin :: "
      + to_string_with_precision( sqrt(pow(this->state.p[0]-this->origin[0],2) +
                                       pow(this->state.p[1]-this->origin[1],2) +
                                       pow(this->state.p[2]-this->origin[2],2)), 4) + " meters"
    << "|" << std::endl;
  std::cout << "| " << std::left << std::setfill(' ') << std::setw(66)
    << "Registration       :: keyframes: " + std::to_string(this->keyframes.size()) + ", "
                               + "deskewed points: " + std::to_string(this->deskew_size)
    << "|" << std::endl;
  std::cout << "|                                                                   |" << std::endl;

  std::cout << std::right << std::setprecision(2) << std::fixed;
  std::cout << "| Computation Time :: "
    << std::setfill(' ') << std::setw(6) << this->comp_times.back()*1000. << " ms    // Avg: "
    << std::setw(6) << avg_comp_time*1000. << " / Max: "
    << std::setw(6) << *std::max_element(this->comp_times.begin(), this->comp_times.end())*1000.
    << "     |" << std::endl;
  std::cout << "| Cores Utilized   :: "
    << std::setfill(' ') << std::setw(6) << (cpu_percent/100.) * this->numProcessors << " cores // Avg: "
    << std::setw(6) << (avg_cpu_usage/100.) * this->numProcessors << " / Max: "
    << std::setw(6) << (*std::max_element(this->cpu_percents.begin(), this->cpu_percents.end()) / 100.)
                       * this->numProcessors
    << "     |" << std::endl;
  std::cout << "| CPU Load         :: "
    << std::setfill(' ') << std::setw(6) << cpu_percent << " %     // Avg: "
    << std::setw(6) << avg_cpu_usage << " / Max: "
    << std::setw(6) << *std::max_element(this->cpu_percents.begin(), this->cpu_percents.end())
    << "     |" << std::endl;
  std::cout << "| " << std::left << std::setfill(' ') << std::setw(66)
    << "RAM Allocation   :: " + to_string_with_precision(resident_set/1000., 2) + " MB"
    << "|" << std::endl;

  std::cout << "+-------------------------------------------------------------------+" << std::endl;

}
