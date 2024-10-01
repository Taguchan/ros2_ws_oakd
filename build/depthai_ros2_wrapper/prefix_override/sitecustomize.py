import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/karin-22/ros2_ws_oakd/install/depthai_ros2_wrapper'
