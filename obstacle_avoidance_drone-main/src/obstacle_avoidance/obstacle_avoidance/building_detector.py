#!/usr/bin/env python3
"""
Downward-facing segmentation node.

Subscribes to the down-camera image topic, runs the SegFormer model
(from inference.py) on the central region of each frame, and publishes:

    /building_below  (std_msgs/Int8)
        1 -> building underneath, do NOT lower altitude
        0 -> ground clear, safe to descend

Inference is throttled to INFER_PERIOD so the GPU keeps up; the newest
camera frame is always used.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Int8
from cv_bridge import CvBridge
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

import cv2
import torch
import numpy as np
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

# =====================================================
# CONFIG
# =====================================================

DOWN_CAMERA_TOPIC = '/world/iris_runway/model/iris_with_gimbal/model/gimbal/link/pitch_link/sensor/camera/image'   # <-- SET to your downward camera topic

MODEL_PATH = "/home/satvik/scp/Segmentation_GATE_1/segformer_b0_best_updated"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

WIDTH  = 1080     # model input resize (same as inference.py)
HEIGHT = 640

BUILDING_CLASS_ID = 1      # UAVid: index of "Building" in CLASS_NAMES
ROI_FRACTION      = 0.5    # central 50% x 50% of the frame = what's under us
BUILDING_THRESH   = 0.05   # >5% building pixels in ROI -> publish 1
INFER_PERIOD      = 0.2    # s -> 5 Hz inference

CLASS_NAMES = [
    "Clutter", "Building", "Road", "Static Car",
    "Tree", "Vegetation", "Human", "Moving Car"
]

# =====================================================


class BuildingDetector(Node):

    def __init__(self):
        super().__init__('building_detector_node')

        self.get_logger().info(f'Loading SegFormer on {DEVICE} ...')
        self.processor = SegformerImageProcessor.from_pretrained("nvidia/mit-b0")
        self.model = SegformerForSemanticSegmentation.from_pretrained(MODEL_PATH)
        self.model.to(DEVICE)
        self.model.eval()
        self.get_logger().info('Model loaded.')

        self.bridge = CvBridge()
        self.latest_frame = None

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1                      # only the newest frame matters
        )
        self.create_subscription(Image, DOWN_CAMERA_TOPIC, self.image_cb, qos)

        self.pub = self.create_publisher(Int8, '/building_below', 10)
        self.create_timer(INFER_PERIOD, self.infer_and_publish)

    def image_cb(self, msg):
        self.latest_frame = msg

    def infer_and_publish(self):
        if self.latest_frame is None:
            # no camera yet -> report 1 (unsafe to descend) so nobody
            # lowers altitude blind
            self.pub.publish(Int8(data=1))
            return

        frame = self.bridge.imgmsg_to_cv2(self.latest_frame, desired_encoding='bgr8')
        frame = cv2.resize(frame, (WIDTH, HEIGHT), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        inputs = self.processor(images=rgb, return_tensors="pt")
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        logits = torch.nn.functional.interpolate(
            outputs.logits, size=(HEIGHT, WIDTH),
            mode="bilinear", align_corners=False,
        )
        pred = logits.argmax(dim=1)[0].cpu().numpy()

        # ── central ROI = the patch of ground directly below the drone ──
        h, w = pred.shape
        rh = int(h * ROI_FRACTION / 2)
        rw = int(w * ROI_FRACTION / 2)
        roi = pred[h // 2 - rh: h // 2 + rh, w // 2 - rw: w // 2 + rw]

        building_frac = float(np.mean(roi == BUILDING_CLASS_ID))
        flag = 1 if building_frac > BUILDING_THRESH else 0

        self.pub.publish(Int8(data=flag))
        self.get_logger().info(
            f'building_below={flag}  (building={building_frac * 100:.1f}% of ROI)',
            throttle_duration_sec=1.0)


def main(args=None):
    rclpy.init(args=args)
    node = BuildingDetector()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
