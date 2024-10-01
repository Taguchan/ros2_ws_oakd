import rclpy
import cv2
import depthai as dai
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

class RgbPreview(Node):
    def __init__(self):
        super().__init__('rgb_preview')
        self.publisher_ = self.create_publisher(Image, '/sensor/depthai/rgb_image', 10)
        timer_period = 0.00001 # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)

        # Create pipeline
        self.pipeline = dai.Pipeline()

        # Define source and output
        camRgb = self.pipeline.create(dai.node.ColorCamera)
        xoutRgb = self.pipeline.create(dai.node.XLinkOut)

        xoutRgb.setStreamName("rgb")

        # Properties
        camRgb.setPreviewSize(224, 224)
        camRgb.setInterleaved(False)
        camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)

        # Linking
        camRgb.preview.link(xoutRgb.input)

        # Connect to device and start pipeline
        self.device = dai.Device(self.pipeline)
        print('Connected cameras:', self.device.getConnectedCameraFeatures())
        print('Usb speed:', self.device.getUsbSpeed().name)
        if self.device.getBootloaderVersion() is not None:
            print('Bootloader version:', self.device.getBootloaderVersion())
        print('Device name:', self.device.getDeviceName(), ' Product name:', self.device.getProductName())

        # Output queue will be used to get the rgb frames from the output defined above
        self.qRgb = self.device.getOutputQueue(name="rgb", maxSize=4, blocking=False)

        # ROS Publisher
        self.bridge = CvBridge()

    def timer_callback(self):
        inRgb = self.qRgb.get()
        cv_frame = inRgb.getCvFrame()
        ros_image_msg = self.bridge.cv2_to_imgmsg(cv_frame, encoding="bgr8")
        self.publisher_.publish(ros_image_msg)


def main(args=None):
    rclpy.init(args=args)
    rgb_preview = RgbPreview()
    rclpy.spin(rgb_preview)

    rgb_preview.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
