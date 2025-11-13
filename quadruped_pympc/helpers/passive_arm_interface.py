import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from std_srvs.srv import Trigger
import serial
import time
import math

class Passive_Arm_Int(Node):
    def __init__(self):
        super().__init__('encoder_publisher')

        # Publisher for joint positions
        self.publisher_ = self.create_publisher(Float32MultiArray, 'arm_joint_positions', 10)

        # Service to set rest position
        self.srv = self.create_service(Trigger, 'set_rest_position', self.set_rest_service)

        # Serial connection to Arduino
        PORT = "/dev/ttyACM0"  # Change for your system
        BAUD = 115200
        self.ser = serial.Serial(PORT, BAUD, timeout=1)

        self.rest_position = None
        self.latest_values = None  # Store latest encoder reading

        # Publish every 50 ms (~20 Hz)
        self.timer = self.create_timer(0.05, self.timer_callback)

    def timer_callback(self):
        """Reads encoder values from Arduino and publishes joint positions."""
        line = self.ser.readline().decode('utf-8').strip()
        if not line:
            return

        try:
            values = [float(v) for v in line.split("\t")]
            self.latest_values = values

            if self.rest_position is None:
                # First time auto-set
                time.sleep(0.1)
                self.rest_position = values[:]
                self.get_logger().info(f"Initial rest position set to: {self.rest_position}")

            # Compute relative values from rest position
            adjusted = []
            for val, offset in zip(values, self.rest_position):
                diff =  offset - val
                # if diff < 0:
                #     diff += 2 * math.pi
                # elif diff >= 2 * math.pi:
                #     diff -= 2 * math.pi
                adjusted.append(diff)
                
            # Check the signs of the adjusted values
            # according to my frame if the angles are between 0 and 2*pi they are positive
            
            msg = Float32MultiArray()
            msg.data = adjusted
            self.publisher_.publish(msg)

        except ValueError:
            self.get_logger().warn(f"Bad data: {line}")

    def set_rest_service(self, request, response):
        """Service callback to update rest position to latest reading."""
        if self.latest_values is not None:
            self.rest_position = self.latest_values[:]
            self.get_logger().info(f"Rest position updated to: {self.rest_position}")
            response.success = True
            response.message = "Rest position set successfully."
        else:
            response.success = False
            response.message = "No encoder data available."
        return response


def main(args=None):
    rclpy.init(args=args)
    node = Passive_Arm_Int()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.ser.close()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
