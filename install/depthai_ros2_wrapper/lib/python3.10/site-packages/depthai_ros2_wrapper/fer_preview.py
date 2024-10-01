import rclpy
import cv2
import depthai as dai
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from fer import FER

class RgbPreviewWithEmotion(Node):
    def __init__(self):
        super().__init__('rgb_preview_with_emotion')
        self.publisher_ = self.create_publisher(Image, '/sensor/depthai/rgb_image_with_emotion', 10)
        timer_period = 0.00001  # seconds
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

        # Load the emotion detection model
        self.emotion_detector = FER(mtcnn=True)

    def timer_callback(self):
        inRgb = self.qRgb.get()
        cv_frame = inRgb.getCvFrame()

        # Perform emotion detection
        result = self.emotion_detector.detect_emotions(cv_frame)

        for face in result:
            (x, y, w, h) = face["box"]
            emotions = face["emotions"]
            emotion, score = max(emotions.items(), key=lambda item: item[1])

            # Draw rectangle around detected face
            cv2.rectangle(cv_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            # Display emotion and score
            cv2.putText(cv_frame, f'{emotion}: {score:.2f}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        # Convert the modified frame to ROS image message
        ros_image_msg = self.bridge.cv2_to_imgmsg(cv_frame, encoding="bgr8")
        self.publisher_.publish(ros_image_msg)


def main(args=None):
    rclpy.init(args=args)
    rgb_preview_with_emotion = RgbPreviewWithEmotion()
    rclpy.spin(rgb_preview_with_emotion)

    rgb_preview_with_emotion.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()