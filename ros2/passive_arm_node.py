#!/usr/bin/env python3

import math
import statistics
from collections import deque

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_srvs.srv import Trigger
import serial


class PassiveArmSerialNode(Node):
    def __init__(self):
        super().__init__('passive_arm_encoder_publisher')

        # ---------------- Parameters ----------------
        self.declare_parameter('port', '/dev/ttyACM0')
        self.declare_parameter('baud', 115200)
        self.declare_parameter('publish_compat_topic', True)

        self.port = self.get_parameter('port').value
        self.baud = int(self.get_parameter('baud').value)
        self.publish_compat_topic = bool(
            self.get_parameter('publish_compat_topic').value
        )

        # These names must match your URDF joint names.
        self.joint_names = [
            'arm_joint1',
            'arm_joint2',
            'arm_joint3'
        ]

        self.num_joints = len(self.joint_names)

        # Direction correction.
        # Your old script inverted joint 1, so this keeps that behavior.
        # Change signs if one joint moves in the wrong direction.
        self.sign = [
            -1.0,   # arm_joint1
             1.0,   # arm_joint2
             1.0    # arm_joint3
        ]

        # Optional static offsets after rest position.
        self.offset = [
            0.0,
            0.0,
            0.0
        ]

        # ---------------- ROS publishers/services ----------------
        # Standard ROS 2 topic used by robot_state_publisher.
        self.joint_state_pub = self.create_publisher(
            JointState,
            'joint_states',
            10
        )

        # Compatibility topic matching your original script.
        self.compat_pub = None
        if self.publish_compat_topic:
            self.compat_pub = self.create_publisher(
                JointState,
                'passive_arm_joint_states',
                10
            )

        self.srv = self.create_service(
            Trigger,
            'set_rest_position',
            self.set_rest_service
        )

        # ---------------- Serial ----------------
        try:
            self.ser = serial.Serial(self.port, self.baud, timeout=1)
            self.ser.reset_input_buffer()
        except Exception as e:
            self.get_logger().error(
                f'Failed to open serial {self.port} @ {self.baud}: {e}'
            )
            raise

        # ---------------- State ----------------
        self.prev_raw = None
        self.continuous_angle = None
        self.latest_continuous_angle = None
        self.rest_position = None

        self.prev_position = None
        self.prev_time_ns = None
        self.prev_velocity = None

        # Filters
        self.position_alpha = 0.3
        self.velocity_alpha = 0.3
        self.min_dt = 1e-4
        self.median_buffers = [
            deque(maxlen=3) for _ in range(self.num_joints)
        ]

        # 20 Hz
        self.period = 0.05
        self.timer = self.create_timer(self.period, self.timer_callback)

        self.get_logger().info(
            f'Passive arm serial node started on {self.port} @ {self.baud}. '
            'Publishing /joint_states.'
        )

    def low_pass_filter(self, new_value, prev_value, alpha):
        return alpha * new_value + (1.0 - alpha) * prev_value

    def parse_serial_line(self, line):
        """
        Expected Arduino output when PLOTTER_MODE=false:

            rad1<TAB>rad2<TAB>rad3

        Example:

            0.123456    1.234567    2.345678
        """
        parts = line.split()

        if len(parts) != self.num_joints:
            raise ValueError(
                f'Expected {self.num_joints} values, got {len(parts)}'
            )

        return [float(v) for v in parts]

    def update_continuous_angles(self, raw_angles):
        """
        Converts absolute 0..2pi encoder angles into continuous angles.
        This prevents jumps when the encoder wraps from 2pi back to 0.
        """
        if self.prev_raw is None:
            self.prev_raw = raw_angles[:]
            self.continuous_angle = raw_angles[:]
            return self.continuous_angle[:]

        for i in range(self.num_joints):
            delta = raw_angles[i] - self.prev_raw[i]

            if delta > math.pi:
                delta -= 2.0 * math.pi
            elif delta < -math.pi:
                delta += 2.0 * math.pi

            self.continuous_angle[i] += delta
            self.prev_raw[i] = raw_angles[i]

        return self.continuous_angle[:]

    def timer_callback(self):
        line = self.ser.readline().decode(
            'utf-8',
            errors='ignore'
        ).strip()

        if not line:
            return

        try:
            raw_angles = self.parse_serial_line(line)
        except Exception as e:
            self.get_logger().warn(f'Bad serial data ({e}): {line}')
            return

        continuous = self.update_continuous_angles(raw_angles)
        self.latest_continuous_angle = continuous[:]

        # First valid sample becomes the initial rest position.
        if self.rest_position is None:
            self.rest_position = continuous[:]
            self.get_logger().info(
                f'Initial rest position set to: {self.rest_position}'
            )
            return

        # Position relative to rest.
        position = []
        for i in range(self.num_joints):
            q = continuous[i] - self.rest_position[i]
            q = self.sign[i] * q + self.offset[i]
            position.append(q)

        # Median filter for single-sample spikes.
        for i in range(self.num_joints):
            self.median_buffers[i].append(position[i])
            position[i] = statistics.median(self.median_buffers[i])

        # Low-pass filter on position.
        if self.prev_position is not None:
            for i in range(self.num_joints):
                position[i] = self.low_pass_filter(
                    position[i],
                    self.prev_position[i],
                    self.position_alpha
                )

        now = self.get_clock().now()

        # Velocity calculation.
        velocity = [0.0] * self.num_joints

        if self.prev_position is not None and self.prev_time_ns is not None:
            dt = (now.nanoseconds - self.prev_time_ns) / 1e9

            if dt >= self.min_dt:
                raw_velocity = [
                    (position[i] - self.prev_position[i]) / dt
                    for i in range(self.num_joints)
                ]

                if self.prev_velocity is None:
                    velocity = raw_velocity
                else:
                    velocity = [
                        self.low_pass_filter(
                            raw_velocity[i],
                            self.prev_velocity[i],
                            self.velocity_alpha
                        )
                        for i in range(self.num_joints)
                    ]

                self.prev_velocity = velocity[:]

        # Build JointState message.
        msg = JointState()
        msg.header.stamp = now.to_msg()
        msg.name = self.joint_names
        msg.position = position
        msg.velocity = velocity

        self.joint_state_pub.publish(msg)

        if self.compat_pub is not None:
            # The controller's legacy compatibility interface expects the
            # measured joint positions followed by the three rest positions.
            # This node reports positions relative to rest, so the reference
            # is the zero vector.
            compat_msg = JointState()
            compat_msg.header = msg.header
            compat_msg.name = self.joint_names + [
                f'{name}_p0' for name in self.joint_names
            ]
            compat_msg.position = position + [0.0] * self.num_joints
            compat_msg.velocity = velocity
            self.compat_pub.publish(compat_msg)

        self.prev_position = position[:]
        self.prev_time_ns = now.nanoseconds

    def set_rest_service(self, request, response):
        """
        Call with:

            ros2 service call /set_rest_position std_srvs/srv/Trigger {}
        """
        if self.latest_continuous_angle is None:
            response.success = False
            response.message = 'No encoder data available yet.'
            return response

        self.rest_position = self.latest_continuous_angle[:]

        # Reset filters to avoid velocity spike after re-zeroing.
        self.prev_position = None
        self.prev_time_ns = None
        self.prev_velocity = None

        self.median_buffers = [
            deque(maxlen=3) for _ in range(self.num_joints)
        ]

        self.get_logger().info(
            f'Rest position updated to: {self.rest_position}'
        )

        response.success = True
        response.message = 'Rest position set successfully.'
        return response

    def destroy_node(self):
        if hasattr(self, 'ser') and self.ser and self.ser.is_open:
            try:
                self.ser.close()
            except Exception:
                pass

        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)

    node = PassiveArmSerialNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
