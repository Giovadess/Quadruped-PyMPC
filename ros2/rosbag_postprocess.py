import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

def read_ros2_bag(bag_path):
    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=bag_path, storage_id='sqlite3'),
        rosbag2_py.ConverterOptions(input_serialization_format='cdr', output_serialization_format='cdr')
    )

    # get topics + types
    topic_types = reader.get_all_topics_and_types()

    def type_for_topic(topic_name):
        for tt in topic_types:
            if tt.name == topic_name:
                return tt.type
        raise KeyError(topic_name)

    while reader.has_next():
        topic, data, timestamp = reader.read_next()
        msg_type = get_message(type_for_topic(topic))
        msg = deserialize_message(data, msg_type)

        # Now you have `msg` — pass to your existing processing
        process_message(topic, msg, timestamp)
def process_message(topic, msg, timestamp):
    # Placeholder function to process each message
    print(f"Topic: {topic}, Timestamp: {timestamp}, Message: {msg}")

if __name__ == "__main__":
    bag_path = "/home/iit.local/gdessy/dls_ws_home/quadruped_pympc_framework/Quadruped-PyMPC/ros2/rosbag_12_12/19_12_rosbags/zmp_trot"
    read_ros2_bag(bag_path)