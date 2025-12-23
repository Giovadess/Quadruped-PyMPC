#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger
import serial
import math
from sensor_msgs.msg import JointState
from collections import deque
from geometry_msgs.msg import Twist

class Passive_Arm_Int(Node):
    def __init__(self):
        super().__init__('encoder_publisher')

        # Publisher for JointState
        self.joint_state_pub = self.create_publisher(JointState, 'passive_arm_joint_states', 10)

        # Service to set rest position (baseline)
        self.srv = self.create_service(Trigger, 'set_rest_position', self.set_rest_service)
        
        self.joint_names = ['arm_joint1', 'arm_joint2', 'arm_joint3']

        # Serial connection to Arduino
        PORT = "/dev/ttyACM0"  # Change for your system
        BAUD = 115200
        self.publish_raw_velocity = False
        try:
            self.ser = serial.Serial(PORT, BAUD, timeout=1)
        except Exception as e:
            self.get_logger().error(f"Failed to open serial {PORT} @ {BAUD}: {e}")
            raise

        # ---------------- State ----------------
        self.rest_position = None             # baseline counts set on first read / service
        self.latest_counts = None             # latest raw counts from serial

        self.prev_pos_rad = None              # previous positions (radians)
        self.prev_time_ns = None

        # Filters
        self.vel_ema = None                   # EMA state (rad/s)
        self.mavg_window = 5                  # moving average window size
        self.use_ema = True
        self.ema_alpha = 0.2                  # EMA alpha
        self.use_mavg = False
        self.min_dt = 1e-4                    # min dt for velocity calc

        self.vel_hist = deque(maxlen=max(2, self.mavg_window))  # history for moving average

        period = 0.05  # 20 Hz
        self.counts_per_rev = 1            # encoder counts per revolution (10-bit)
        self.count_to_rad = (2.0 * math.pi) / self.counts_per_rev  # counts -> radians
        self.count_to_rad= 1


        # ---------------- Loop ----------------
        self.timer = self.create_timer(period, self.timer_callback)
        self.get_logger().info(
            "PassiveArmInterface started. JointState in radians / radians per second. "
            f"counts_per_rev={self.counts_per_rev}, EMA={self.use_ema} (alpha={self.ema_alpha}), "
            f"MAVG={self.use_mavg} (win={self.mavg_window})."
        )

    def timer_callback(self):
        """Reads encoder values from Arduino and publishes joint positions."""
        line = self.ser.readline().decode('utf-8', errors='ignore').strip()
        if not line:
            return

        try:
            values = [float(v) for v in line.split("\t")]
        except Exception as e:
            self.get_logger().warn(f"Bad serial data ({e}): {line}")
            return

        # Expect exactly 3 values (one per joint)
        if len(values) != len(self.joint_names):
            self.get_logger().warn(f"Expected {len(self.joint_names)} values, got {len(values)}: {line}")
            return

        self.latest_counts = values

        # First time: set rest baseline and wait for next sample to compute velocity
        if self.rest_position is None:
            self.rest_position = values[:]
            self.get_logger().info(f"Initial rest position (counts) set to: {self.rest_position}")
            return

        # ---- Counts → radians (relative to rest) ----
        # position_rad = (rest - current) * (2π/CPR)
        pos_rad = [(c - r) * self.count_to_rad for c, r in zip(values, self.rest_position)]

        # ---- Build JointState ----
        now = self.get_clock().now()
        msg = JointState()
        msg.header.stamp = now.to_msg()
        msg.name = self.joint_names
        msg.position = pos_rad

        # ---- Velocity (finite difference → filters) ----
        if self.prev_pos_rad is not None and self.prev_time_ns is not None:
            dt = (now.nanoseconds - self.prev_time_ns) / 1e9
            if dt >= self.min_dt:
                v_raw = [(c - p) / dt for c, p in zip(pos_rad, self.prev_pos_rad)]  # rad/s

                v_filt = v_raw
                # EMA
                if self.use_ema:
                    if self.vel_ema is None:
                        self.vel_ema = v_raw[:]  # seed
                    else:
                        a = self.ema_alpha
                        self.vel_ema = [a * n + (1.0 - a) * o for n, o in zip(v_raw, self.vel_ema)]
                    v_filt = self.vel_ema

                # Moving average on top of EMA (optional)
                if self.use_mavg and self.mavg_window >= 2:
                    self.vel_hist.append(v_filt[:])
                    n = len(self.vel_hist)
                    accum = [0.0] * len(v_filt)
                    for vec in self.vel_hist:
                        for i, val in enumerate(vec):
                            accum[i] += val
                    v_filt = [s / n for s in accum]

                msg.velocity = v_filt  # rad/s
        ### Compute and publish follower velocity
        

        # Publish
        self.joint_state_pub.publish(msg)

        # Update state for next tick
        self.prev_pos_rad = pos_rad[:]
        self.prev_time_ns = now.nanoseconds

    def set_rest_service(self, request, response):
        """Update rest position to latest reading (values) only."""
        if self.latest_counts is not None:
            self.rest_position = self.latest_counts[:]
            self.get_logger().info(f"Rest position updated to (counts): {self.rest_position}")
            response.success = True
            response.message = "Rest position set successfully."
        else:
            response.success = False
            response.message = "No encoder data available."
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
    node = Passive_Arm_Int()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
