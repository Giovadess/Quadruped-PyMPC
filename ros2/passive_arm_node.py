#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger
import serial
import math
from sensor_msgs.msg import JointState
from collections import deque
from geometry_msgs.msg import Twist
import statistics # <--- ADD THIS
from collections import deque
class Passive_Arm_Int(Node):
    def __init__(self):
        super().__init__('encoder_publisher')

        # Publisher for JointState
        self.joint_state_pub = self.create_publisher(JointState, 'passive_arm_joint_states', 10)

        # Service to set rest position (baseline)
        self.srv = self.create_service(Trigger, 'set_rest_position', self.set_rest_service)
        
        self.joint_names = ['arm_joint1', 'arm_joint2', 'arm_joint3','arm_joint1_p0','arm_joint2_p0','arm_joint3_p0']
        

        # Serial connection to Arduino
        PORT = "/dev/ttyACM0"  # Change for your system
        BAUD = 115200
        self.publish_raw_velocity = False
        self.rest_position_set = False
        try:
            self.ser = serial.Serial(PORT, BAUD, timeout=1)
        except Exception as e:
            self.get_logger().error(f"Failed to open serial {PORT} @ {BAUD}: {e}")
            raise

        # ---------------- State ----------------
        self.rest_position = None             # baseline counts set on first read / service
        self.latest_counts = None             # latest raw counts from serial

        self.start_position =   None             # initial position at startup (counts)

        self.prev_pos_rad = None              # previous positions (radians)
        self.prev_time_ns = None

        # Filters
        self.vel_ema = None                   # EMA state (rad/s)
        self.mavg_window = 5                  # moving average window size
        self.use_ema = True
        self.ema_alpha = 0.2                  # EMA alpha
        self.use_mavg = False
        self.min_dt = 1e-4                    # min dt for velocity calc

        self.previous_velocity = None         # for low-pass filtering of velocity
        self.vel_hist = deque(maxlen=max(2, self.mavg_window))  # history for moving average

        period = 0.05  # 20 Hz
        self.counts_per_rev = 1024            # encoder counts per revolution (10-bit)
        self.count_to_rad = (2.0 * math.pi) / self.counts_per_rev  # counts -> radians
        # ... inside __init__ ...
        self.prev_pos_rad = None 
        self.prev_time_ns = None
        # ADD THIS BLOCK:
        # 3 buffers (one per joint), window size 3 is perfect for single spikes
        self.median_buffers = [deque(maxlen=3) for _ in range(len(self.joint_names)//2)]
        # ---------------- Loop ----------------
        self.timer = self.create_timer(period, self.timer_callback)
        self.get_logger().info(
            "PassiveArmInterface started. JointState in radians / radians per second. "
            f"counts_per_rev={self.counts_per_rev}, EMA={self.use_ema} (alpha={self.ema_alpha}), "
            f"MAVG={self.use_mavg} (win={self.mavg_window})."
        )

    def unwrap(self, theta, theta_ref):
        d = theta - theta_ref
        return theta_ref + ((d + math.pi) % (2*math.pi) - math.pi)
    
    def low_pass_filter(self, new_value, prev_value, alpha):
        return alpha * new_value + (1 - alpha) * prev_value

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
        if len(values) != len(self.joint_names)/2:
            self.get_logger().warn(f"Expected {len(self.joint_names)} values, got {len(values)}: {line}")
            return

        self.latest_counts = values

        # First time: set rest baseline and wait for next sample to compute velocity
        if self.start_position is None:
            self.start_position = values[:]
            self.get_logger().info(f"Initial rest position (counts) set to: {self.start_position}")
            ## At this first call also set the rest position
            self.rest_position = values[:]
            self.get_logger().info(f"Rest position (counts) set to: {self.rest_position}")
            return
        


        # ---- Counts → radians (relative to rest) ----
        # position_rad = (rest - current) * (2π/CPR)
        # pos_rad = [(c - r) * self.count_to_rad for c, r in zip(values, self.rest_position)]

        ###

        # pos_rad =      [c * self.count_to_rad for c in values]
        # pos_rest_rad = [r * self.count_to_rad for r in self.rest_position]

        # curr_pos = [ c-r for c,r in zip(pos_rad,pos_rest_rad)]

        ## Save the initial position (not rest but the actual position at startup)
        ## the joint position is computed as the difference between the current position and the start position
        ## its already given in radians
        # q_pos = [c for c in values]
        # q_pos_rest = [r for r in self.start_position]


        theta_raw   = values
        theta_start = self.start_position
        theta_rest  = self.rest_position

        # theta_cont = [
        #     t0 + ((t - t0 + math.pi) % (2*math.pi) - math.pi)
        #     for t, t0 in zip(theta_raw, theta_start)
        # ]

        # q_pos_rest = [tc - tr for tc, tr in zip(theta_cont, theta_rest)]
        # q_pos = [tc - ts for tc, ts in zip(theta_cont, theta_start)]



        theta_cont = [
            self.unwrap(t, ts)
            for t, ts in zip(theta_raw, theta_start)
        ]

        theta_rest_cont = [
            self.unwrap(tr, ts)
            for tr, ts in zip(theta_rest, theta_start)
        ]

        q_pos      = [tc - ts for tc, ts in zip(theta_cont, theta_start)]
        q_pos_rest = [trc - ts for trc, ts in zip(theta_rest_cont, theta_start)]
        
        q_pos[0] = -q_pos[0]
        q_pos_rest[0] = -q_pos_rest[0]

        ## Add a spike filter to avoid jumps due to unwrapping
        for i in range(len(q_pos)):
            if abs(q_pos[i] - q_pos_rest[i]) > math.pi:
                q_pos[i] = q_pos_rest[i]

        # ----------------- FILTERING SECTION -----------------
        for i in range(len(q_pos)):
            # 1. Feed the raw value into the history buffer
            self.median_buffers[i].append(q_pos[i])

            # 2. MEDIAN FILTER: Pick the middle value (ignores the spike)
            # If buffer isn't full yet, just use current value to avoid errors
            if len(self.median_buffers[i]) > 0:
                clean_val = statistics.median(self.median_buffers[i])
            else:
                clean_val = q_pos[i]


        # Filtering to avoid jumps due to unwrapping

        if self.prev_pos_rad is not None:
            for i in range(len(q_pos)):
                q_pos[i] = self.low_pass_filter(q_pos[i], self.prev_pos_rad[i], 0.3)


        # ---- Build JointState ----
        now = self.get_clock().now()
        msg = JointState()
        msg.header.stamp = now.to_msg()
        msg.name = ['arm_joint1', 'arm_joint2', 'arm_joint3','arm_joint1_p0','arm_joint2_p0','arm_joint3_p0']

        # msg.position = curr_pos + pos_rest_rad  # radians

        msg.position = q_pos + q_pos_rest  # radians

        # ---- Velocity (finite difference → filters) ----
        if self.prev_pos_rad is not None and self.prev_time_ns is not None:
            dt = (now.nanoseconds - self.prev_time_ns) / 1e9
            if dt >= self.min_dt:
                v_raw = [(c - p) / dt for c, p in zip(q_pos, self.prev_pos_rad)]  # rad/s

                v_filt = v_raw
            ## filter velocity with low-pass to avoid jumps

                if self.previous_velocity is None:
                    self.previous_velocity = v_filt[:]
                else:
                    alpha = 0.3
                    v_filt = [
                        self.low_pass_filter(v, pv, alpha)
                        for v, pv in zip(v_filt, self.previous_velocity)
                    ]
                    self.previous_velocity = v_filt[:]
                
                msg.velocity = v_filt 
                # EMA
                # if self.use_ema:
                #     if self.vel_ema is None:
                #         self.vel_ema = v_raw[:]  # seed
                #     else:
                #         a = self.ema_alpha
                #         self.vel_ema = [a * n + (1.0 - a) * o for n, o in zip(v_raw, self.vel_ema)]
                #     v_filt = self.vel_ema

                # # Moving average on top of EMA (optional)
                # if self.use_mavg and self.mavg_window >= 2:
                #     self.vel_hist.append(v_filt[:])
                #     n = len(self.vel_hist)
                #     accum = [0.0] * len(v_filt)
                #     for vec in self.vel_hist:
                #         for i, val in enumerate(vec):
                #             accum[i] += val
                #     v_filt = [s / n for s in accum]

                
        ### Compute and publish follower velocity
        # ## Filter velocity with low-pass to avoid jumps
        # if self.previous_velocity is None:
        #     self.previous_velocity = msg.velocity[:]
        # else:
        #     alpha = 0.3
        #     msg.velocity = [
        #         self.low_pass_filter(v, pv, alpha)
        #         for v, pv in zip(msg.velocity, self.previous_velocity)
        #     ]
        #     self.previous_velocity = msg.velocity[:]
        
        
        # Publish
        self.joint_state_pub.publish(msg)

        # Update state for next tick
        self.prev_pos_rad = q_pos[:]
        self.prev_time_ns = now.nanoseconds

    def set_rest_service(self, request, response):
        """Update rest position to latest reading (values) only."""
        if self.latest_counts is not None:
            self.rest_position = self.latest_counts[:]
            self.rest_position_set = True
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
