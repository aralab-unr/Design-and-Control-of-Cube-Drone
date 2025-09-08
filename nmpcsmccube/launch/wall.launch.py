import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
import xacro

def generate_launch_description():
    # Get package share directory
    package_share_directory = get_package_share_directory('nmpcsmccube')
    
    # Path to the world file
    world_file_path = os.path.join(package_share_directory, 'worlds', 'default.world')

    # Declare world launch argument
    declare_world_cmd = DeclareLaunchArgument(
        'world',
        default_value=world_file_path,
        description='Full path to the world model'
    )

    # Include the Gazebo launch file with the world argument
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('gazebo_ros'),
                'launch',
                'gazebo.launch.py'
            )
        ),
        launch_arguments={'world': LaunchConfiguration('world')}.items()
    )

    # Get the path to the UAV URDF file
    urdf_file_path = os.path.join(package_share_directory, 'urdf', 'cube.urdf.xacro')

    # Convert Xacro to URDF
    doc = xacro.process_file(urdf_file_path)
    urdf_xml = doc.toprettyxml(indent='  ')

    # Define the robot_state_publisher node
    node_robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[{'robot_description': urdf_xml}]
    )

    # Define the spawn_entity node
    spawn_entity = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        name='spawn_entity',
        output='screen',
        arguments=[
            '-topic', 'robot_description',
            '-entity', 'cube',
            '-x', '0', '-y', '0', '-z', '0.1775'
        ]
    )

    # Return the launch description
    return LaunchDescription([
        declare_world_cmd,
        gazebo,
        node_robot_state_publisher,
        spawn_entity
    ])
