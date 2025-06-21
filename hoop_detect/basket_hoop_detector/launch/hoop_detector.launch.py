from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='basket_hoop_detector',
            executable='hoop_detector_node',
            name='hoop_detector',
            parameters=[{
                'debug_level': 1,
                'hoop_radius': 0.23,
                'config_file': 'config/hsv_config.yaml',  # 路径可改为绝对路径
                # camera_matrix and dist_coeffs can be provided here or through YAML
            }]
        )
    ]) 