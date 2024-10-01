import rclpy
import cv2
import depthai as dai
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

class StereoPreview(Node):
    def __init__(self):
        super().__init__('stereo_preview')
        self.left_publisher_ = self.create_publisher(Image, '/sensor/depthai/left_image', 10)
        self.right_publisher_ = self.create_publisher(Image, '/sensor/depthai/right_image', 10)
        timer_period = 0.00001 # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.bridge = CvBridge()

        # Create pipeline
        self.pipeline = dai.Pipeline()

        # Define source and output
        monoLeft = self.pipeline.create(dai.node.MonoCamera)
        monoRight = self.pipeline.create(dai.node.MonoCamera)
        xoutLeft = self.pipeline.create(dai.node.XLinkOut)
        xoutRight = self.pipeline.create(dai.node.XLinkOut)

        xoutLeft.setStreamName('left')
        xoutRight.setStreamName('right')

        # Properties
        monoLeft.setCamera("left")
        monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_480_P)
        monoRight.setCamera("right")
        monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_480_P)

        # Linking
        monoRight.out.link(xoutRight.input)
        monoLeft.out.link(xoutLeft.input)

        # Connect to device and start pipeline
        self.device = dai.Device(self.pipeline)
        print('Connected cameras:', self.device.getConnectedCameraFeatures())
        print('Usb speed:', self.device.getUsbSpeed().name)
        if self.device.getBootloaderVersion() is not None:
            print('Bootloader version:', self.device.getBootloaderVersion())
        print('Device name:', self.device.getDeviceName(), ' Product name:', self.device.getProductName())

        # Output queues will be used to get the grayscale frames from the outputs defined above
        self.qLeft = self.device.getOutputQueue(name="left", maxSize=4, blocking=False)
        self.qRight = self.device.getOutputQueue(name="right", maxSize=4, blocking=False)


    def timer_callback(self):
        inLeft = self.qLeft.get()
        inRight = self.qRight.get()
        cv_left_frame = inLeft.getCvFrame()
        cv_right_frame = inRight.getCvFrame()
        ros_left_image_msg = self.bridge.cv2_to_imgmsg(cv_left_frame, encoding="mono8")
        ros_right_image_msg = self.bridge.cv2_to_imgmsg(cv_right_frame, encoding="mono8")
        self.left_publisher_.publish(ros_left_image_msg)
        self.right_publisher_.publish(ros_right_image_msg)



def main(args=None):
    rclpy.init(args=args)
    stereo_preview = StereoPreview()
    rclpy.spin(stereo_preview)

    stereo_preview.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
