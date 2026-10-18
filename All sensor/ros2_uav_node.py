#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
import numpy as np
import onnxruntime as ort
import math
import time

# PX4 / ROS 2 Message Definitions (Standard px4_msgs interfaces)
try:
    from px4_msgs.msg import (
        VehicleOdometry,
        BatteryStatus,
        TrajectorySetpoint,
        OffboardControlMode,
        VehicleCommand
    )
except ImportError:
    # Mock fallback for non-ROS local testing environments
    VehicleOdometry = BatteryStatus = TrajectorySetpoint = OffboardControlMode = VehicleCommand = object

class UAVTrajectoryTrackingNode(Node):
    def __init__(self):
        super().__init__('uav_trajectory_tracking_node')

        # Declare parameters
        self.declare_parameter('onnx_model_path', 'checkpoints/onnx/energy_surrogate.onnx')
        self.declare_parameter('cruise_speed_ms', 18.05)
        self.declare_parameter('target_altitude_m', 25.0)

        model_path = self.get_parameter('onnx_model_path').get_parameter_value().string_value
        self.cruise_speed = self.get_parameter('cruise_speed_ms').get_parameter_value().double_value
        self.target_alt = self.get_parameter('target_altitude_m').get_parameter_value().double_value

        # Initialize ONNX Runtime Inference Session
        self.get_logger().info(f"Loading ONNX Energy Surrogate from {model_path}...")
        try:
            self.ort_session = ort.InferenceSession(model_path)
            self.input_name = self.ort_session.get_inputs()[0].name
        except Exception as e:
            self.get_logger().warn(f"ONNX model load failed ({e}). Operating in pass-through mode.")
            self.ort_session = None

        # QoS Profiles matching PX4 DDS conventions
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # Subscribers
        self.odom_sub = self.create_subscription(
            VehicleOdometry,
            '/fmu/out/vehicle_odometry',
            self.odometry_callback,
            qos_profile
        )
        self.battery_sub = self.create_subscription(
            BatteryStatus,
            '/fmu/out/battery_status',
            self.battery_callback,
            qos_profile
        )

        # Publishers
        self.setpoint_pub = self.create_publisher(
            TrajectorySetpoint,
            '/fmu/in/trajectory_setpoint',
            qos_profile
        )
        self.offboard_mode_pub = self.create_publisher(
            OffboardControlMode,
            '/fmu/in/offboard_control_mode',
            qos_profile
        )
        self.vehicle_cmd_pub = self.create_publisher(
            VehicleCommand,
            '/fmu/in/vehicle_command',
            qos_profile
        )

        # State Variables
        self.current_pos = np.array([0.0, 0.0, 0.0]) # NED (x, y, z)
        self.current_vel = np.array([0.0, 0.0, 0.0])
        self.battery_soc = 1.0
        self.current_payload_kg = 1.85
        
        # Waypoints in Local NED (x: North, y: East, z: Down)
        self.waypoint_queue = [
            np.array([0.0, 0.0, -self.target_alt]),
            np.array([150.0, 200.0, -self.target_alt]),
            np.array([300.0, -100.0, -self.target_alt]),
            np.array([0.0, 0.0, -self.target_alt])
        ]
        self.active_wp_idx = 0

        # 20 Hz Control & Setpoint Timer
        self.timer = self.create_timer(0.05, self.control_loop)
        self.get_logger().info("UAV Trajectory Tracking & DL Surrogate Node Online.")

    def odometry_callback(self, msg: VehicleOdometry):
        self.current_pos = np.array([msg.position[0], msg.position[1], msg.position[2]])
        self.current_vel = np.array([msg.velocity[0], msg.velocity[1], msg.velocity[2]])

    def battery_callback(self, msg: BatteryStatus):
        self.battery_soc = msg.remaining

    def predict_energy_consumption(self, ground_speed: float, wind_speed: float = 3.0) -> float:
        """Executes real-time forward pass on ONNX surrogate engine."""
        if not self.ort_session:
            return 280.0
        
        feature_vec = np.array([[
            ground_speed,
            self.current_payload_kg,
            wind_speed,
            0.52,  # Relative wind heading (radians)
            0.0    # Delta altitude
        ]], dtype=np.float32)

        outputs = self.ort_session.run(None, {self.input_name: feature_vec})
        return float(outputs[0][0][0])

    def publish_offboard_heartbeat(self):
        msg = OffboardControlMode()
        msg.position = True
        msg.velocity = False
        msg.acceleration = False
        msg.attitude = False
        msg.body_rate = False
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.offboard_mode_pub.publish(msg)

    def control_loop(self):
        # 1. Maintain offboard mode watchdog
        self.publish_offboard_heartbeat()

        if self.active_wp_idx >= len(self.waypoint_queue):
            return

        target_wp = self.waypoint_queue[self.active_wp_idx]
        tracking_error = np.linalg.norm(self.current_pos - target_wp)

        # 2. Check waypoint arrival threshold (2.0 meters)
        if tracking_error < 2.0:
            self.get_logger().info(f"Target Waypoint {self.active_wp_idx + 1} Reached. Advancing...")
            self.active_wp_idx += 1
            if self.active_wp_idx >= len(self.waypoint_queue):
                self.get_logger().info("Mission Complete. Executing auto-land.")
                return

        # 3. Compute surrogate power consumption
        speed_magnitude = float(np.linalg.norm(self.current_vel))
        predicted_power = self.predict_energy_consumption(speed_magnitude)

        # 4. Stream Trajectory Setpoint
        setpoint = TrajectorySetpoint()
        setpoint.position = [float(target_wp[0]), float(target_wp[1]), float(target_wp[2])]
        setpoint.yaw = math.atan2(target_wp[1] - self.current_pos[1], target_wp[0] - self.current_pos[0])
        setpoint.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.setpoint_pub.publish(setpoint)

def main(args=None):
    rclpy.init(args=args)
    node = UAVTrajectoryTrackingNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()