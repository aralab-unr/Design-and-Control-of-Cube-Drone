#
#   Copyright (c)     
#
#   The Verifiable & Control-Theoretic Robotics (VECTR) Lab
#   University of California, Los Angeles
#
#   Authors: Kenny J. Chen, Ryan Nemiroff, Brett T. Lopez
#   Contact: {kennyjchen, ryguyn, btlopez}@ucla.edu
#

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition   
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    current_pkg = FindPackageShare('lio_imm')

    # Set default arguments
    rviz = LaunchConfiguration('rviz', default='true')
    pointcloud_topic = LaunchConfiguration('pointcloud_topic', default='points_raw')
    imu_topic = LaunchConfiguration('imu_topic', default='imu_raw')
    img_topic = LaunchConfiguration('img_topic', default='image_raw')

    # Define arguments
    declare_rviz_arg = DeclareLaunchArgument(
        'rviz',
        default_value=rviz,
        description='Launch RViz'
    )
    declare_pointcloud_topic_arg = DeclareLaunchArgument(
        'pointcloud_topic',
        default_value=pointcloud_topic,
        description='Pointcloud topic name'
    )
    declare_imu_topic_arg = DeclareLaunchArgument(
        'imu_topic',
        default_value=imu_topic,
        description='IMU topic name'
    )

    declare_img_topic_arg = DeclareLaunchArgument(
        'img_topic',
        default_value=img_topic,
        description='IMAGE topic name'
    )

    # Load parameters
    liom_yaml_path = PathJoinSubstitution([current_pkg, 'cfg', 'liom.yaml'])
    liom_params_yaml_path = PathJoinSubstitution([current_pkg, 'cfg', 'params.yaml'])

    # liom Odometry Node
    liom_odom_node = Node(
        package='lio_imm',
        executable='liom_odom_node',
        # executable='liom_odom_node_test',
        # executable='liom_odom_node_test_ver2',
        output='screen',
        parameters=[liom_yaml_path, liom_params_yaml_path],
        remappings=[
            ('pointcloud', pointcloud_topic),
            ('imu', imu_topic),
            ('image', img_topic),
            ('odom', 'liom/odom_node/odom'),
            ('pose', 'liom/odom_node/pose'),
            ('path', 'liom/odom_node/path'),
            ('kf_pose', 'liom/odom_node/keyframes'),
            ('kf_cloud', 'liom/odom_node/pointcloud/keyframe'),
            ('deskewed', 'liom/odom_node/pointcloud/deskewed'),
        ],
    )

    # liom Mapping Node
    liom_map_node = Node(
        package='lio_imm',
        executable='liom_map_node',
        output='screen',
        parameters=[liom_yaml_path, liom_params_yaml_path],
        remappings=[
            ('keyframes', 'liom/odom_node/pointcloud/keyframe'),
        ],
    )

    # RViz node
    rviz_config_path = PathJoinSubstitution([current_pkg, 'launch', 'liom.rviz'])
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='liom_rviz',
        arguments=['-d', rviz_config_path],
        output='screen',
        condition=IfCondition(LaunchConfiguration('rviz'))
    )

    return LaunchDescription([
        declare_rviz_arg,
        declare_pointcloud_topic_arg,
        declare_imu_topic_arg,
        liom_odom_node,
        liom_map_node,
        rviz_node
    ])
