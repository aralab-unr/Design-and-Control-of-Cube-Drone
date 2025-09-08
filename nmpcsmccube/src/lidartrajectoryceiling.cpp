#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>
#include "nav_msgs/msg/odometry.hpp"
#include "tf2/LinearMath/Quaternion.h"
#include "tf2/LinearMath/Matrix3x3.h"
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <vector>
#include <memory>

class Trajectoryprocessor : public rclcpp::Node
{
public:
  Trajectoryprocessor(): Node("point_cloud_to_trajectory"), xs(0), ys(0), zs(0), xds(0), yds(0), zds(0), phis(0), thetas(0), psis(0), phids(0), thetads(0), psids(0) {
    state_sub_ = this->create_subscription<nav_msgs::msg::Odometry>("/droneposition/odom", 10, std::bind(&Trajectoryprocessor::stateCallback, this, std::placeholders::_1));
    trajectory_pub_ = this->create_publisher<std_msgs::msg::Float64MultiArray>("/trajectory", 10);
    point_cloud_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>("/lidar1/out", 10, std::bind(&Trajectoryprocessor::pointCloudCallback, this, std::placeholders::_1));
  }

private:

  void stateCallback(const nav_msgs::msg::Odometry::SharedPtr msg) {

    const auto &position = msg->pose.pose.position;
    const auto &orientation = msg->pose.pose.orientation;
    const auto &linear_velocity = msg->twist.twist.linear;
    const auto &angular_velocity = msg->twist.twist.angular;

    tf2::Quaternion q(orientation.x, orientation.y, orientation.z, orientation.w);
    tf2::Matrix3x3 rpy(q);
    double roll, pitch, yaw;
    rpy.getRPY(roll, pitch, yaw);

    xs = position.x;
    ys = position.y;
    zs = position.z;
    phis = roll;  
    thetas = pitch;  
    psis = yaw;  
    xds = linear_velocity.x;
    yds = linear_velocity.y;
    zds = linear_velocity.z;
    phids = angular_velocity.x;
    thetads = angular_velocity.y;
    psids = angular_velocity.z;
  }

  void pointCloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
  {
      pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
      pcl::fromROSMsg(*msg, *cloud);

      double sum_x = 0.0;
      for (const auto& point : cloud->points)
      {
          sum_x += point.x;
      }

      double range_ceiling = (cloud->points.empty()) ? 0.0 : sum_x / cloud->points.size();

      RCLCPP_INFO(this->get_logger(), "Range: %f", range_ceiling);
      std_msgs::msg::Float64MultiArray trajectory_msg;
      trajectory_msg.data = {0, 0, zs + range_ceiling - 0.1};
      trajectory_pub_->publish(trajectory_msg);
  }
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr point_cloud_sub_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr state_sub_; 
  rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr trajectory_pub_;   
  double xs, ys, zs, xds, yds, zds;
  double phis, thetas, psis, phids, thetads, psids;
};

int main(int argc, char **argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<Trajectoryprocessor>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}