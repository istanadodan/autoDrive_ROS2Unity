from ultralytics import YOLO
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from pathlib import Path
from yolov8_msgs.msg import InferenceResult
from yolov8_msgs.msg import Yolov8Inference
from ament_index_python.packages import get_package_share_directory

bridge = CvBridge()


class Camera_subscriber(Node):

    def __init__(self):
        super().__init__("object_detection_node")

        self.model = YOLO(
            Path(
                get_package_share_directory("object_detection"),
                "resource/model/yolov8n.pt",
            )
        )

        self.yolov8_inference = Yolov8Inference()

        self.subscription = self.create_subscription(
            Image, "jetauto/detp_camera/img_raw", self.camera_callback, 10
        )

        self.yolov8_pub = self.create_publisher(Yolov8Inference, "/Yolov8_Inference", 1)
        self.img_pub = self.create_publisher(Image, "/inference_result", 1)

    def camera_callback(self, data):

        img = bridge.imgmsg_to_cv2(data, "bgr8")
        results = self.model(img)

        self.yolov8_inference.header.frame_id = "inference"
        # self.yolov8_inference.header.stamp = camera_subscriber.get_clock().now().to_msg()
        self.yolov8_inference.header.stamp = self.get_clock().now().to_msg()

        for r in results:
            boxes = r.boxes
            for box in boxes:
                self.inference_result = InferenceResult()
                b = (
                    box.xyxy[0].to("cpu").detach().numpy().copy()
                )  # get box coordinates in (top, left, bottom, right) format
                c = box.cls
                self.inference_result.class_name = self.model.names[int(c)]
                self.inference_result.top = int(b[0])
                self.inference_result.left = int(b[1])
                self.inference_result.bottom = int(b[2])
                self.inference_result.right = int(b[3])
                self.yolov8_inference.yolov8_inference.append(self.inference_result)

            self.get_logger().info(f"{self.yolov8_inference}")

        annotated_frame = results[0].plot()
        img_msg = bridge.cv2_to_imgmsg(annotated_frame)

        self.img_pub.publish(img_msg)
        self.yolov8_pub.publish(self.yolov8_inference)
        self.yolov8_inference.yolov8_inference.clear()


def main(args=None):
    rclpy.init(args=args)
    camera_subscriber = Camera_subscriber()
    rclpy.spin(camera_subscriber)
    rclpy.shutdown()
