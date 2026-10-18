import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    # Package Paths
    pkg_share = os.getcwd()
    
    # Launch Configuration Flags
    use_sim_time = LaunchConfiguration('use_sim_time', default='true')
    enable_gui = LaunchConfiguration('gui', default='true')
    model_path = LaunchConfiguration('model_path', default='checkpoints/onnx/energy_surrogate.onnx')

    # 1. Micro-Climate Wind Field Publisher
    wind_publisher_node = Node(
        package='uav_control_bridge',
        executable='weather_collector.py',
        name='micro_climate_publisher',
        parameters=[{
            'use_sim_time': use_sim_time,
            'poll_interval_sec': 5.0,
            'sim_wind_magnitude_ms': 3.5
        }],
        output='screen'
    )

    # 2. Companion Onboard Trajectory Tracking Node
    trajectory_tracker_node = Node(
        package='uav_control_bridge',
        executable='ros2_uav_node.py',
        name='trajectory_tracking_node',
        parameters=[{
            'use_sim_time': use_sim_time,
            'onnx_model_path': model_path,
            'cruise_speed_ms': 18.05,
            'target_altitude_m': 25.0
        }],
        output='screen'
    )

    # 3. Dynamic Airspace Geofence Monitor Node
    geofence_monitor_node = Node(
        package='uav_control_bridge',
        executable='spatial_geometry.py',
        name='airspace_geofence_monitor',
        parameters=[{
            'use_sim_time': use_sim_time,
            'geojson_airspace_file': 'data/singapore_airspace.geojson'
        }],
        output='screen'
    )

    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='true', description='Use simulation clock'),
        DeclareLaunchArgument('gui', default_value='true', description='Launch Gazebo visualization GUI'),
        DeclareLaunchArgument('model_path', default_value='checkpoints/onnx/energy_surrogate.onnx', description='ONNX model path'),
        
        wind_publisher_node,
        trajectory_tracker_node,
        geofence_monitor_node
    ])