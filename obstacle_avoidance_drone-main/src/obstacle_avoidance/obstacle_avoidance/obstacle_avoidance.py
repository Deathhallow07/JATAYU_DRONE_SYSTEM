import rclpy
import numpy as np
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Int8
from cv_bridge import CvBridge
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
import cv2
import socket
import json
import time
import math


# ── AVOIDANCE CONFIG ─────────────────────────────────────────────────────────
AVOID_HOST     = '127.0.0.1'
AVOID_CMD_PORT = 9100    # node -> drone1: {"vz": ..., "hold": ...}
AVOID_ALT_PORT = 9101    # drone1 -> node: {"alt": m, "mission_alt": m}
CLIMB_SPEED    = 2.0     # m/s upward
DESCEND_SPEED  = 1.0     # m/s downward (constant, while moving forward)
OBSTACLE_FAR   = 5.0     # m — climb-while-moving band starts here
OBSTACLE_NEAR  = 3.0     # m — stop-and-climb-only below this
EXTRA_CLIMB    = 6.0     # m — measured altitude gain required after clearing
ALT_TOL        = 0.3     # m — "back at mission altitude" tolerance
BELOW_STALE    = 1.0     # s — /building_below older than this => treat as 1
EXTRA_CLIMB_MAX_TIME = 12.0   # s — safety backstop ONLY (alt feedback dead)
# ─────────────────────────────────────────────────────────────────────────────


class Avoider(Node):

    def __init__(self):
        super().__init__('obstacle_avoidance_node')

        self.range_min = 1.0
        self.range_max = 5.0
        self.bridge = CvBridge()
        self.target = None

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        self.create_subscription(
            Image,
            '/world/iris_runway/model/iris_with_gimbal/model/gimbal/link/pitch_link/sensor/camera_2/depth_image',
            self.DepthcamCallback,
            qos
        )
        self.DepthcamMessage = None

        # ── downward building detector (from building_detector.py) ──
        self.create_subscription(Int8, '/building_below', self.BuildingBelowCallback, 10)
        self.building_below = 1        # default: assume unsafe until told otherwise
        self.building_below_t = 0.0    # last message time (staleness check)

        # ── UDP command channel to drone1_fullgcs.py (send) ──
        self.cmd_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.cmd_addr = (AVOID_HOST, AVOID_CMD_PORT)

        # ── UDP altitude feedback from drone1_fullgcs.py (receive) ──
        self.alt_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.alt_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.alt_sock.bind((AVOID_HOST, AVOID_ALT_PORT))
        self.alt_sock.setblocking(False)
        self.current_alt = None        # latest measured relative altitude (m)
        self.mission_alt = None        # target cruise altitude, sent by drone1

        # ── avoidance state ──
        self.state = 'SEARCH'          # SEARCH -> CLIMB -> EXTRA_CLIMB -> SEARCH
        self.extra_start_alt = None    # altitude when roofline cleared
        self.extra_start_time = None   # for the safety backstop only

        self.controller_timer = self.create_timer(0.1, self.Controller)

    def DepthcamCallback(self, msg):
        self.DepthcamMessage = msg

    def BuildingBelowCallback(self, msg):
        self.building_below = int(msg.data)
        self.building_below_t = time.time()

    def _ground_clear(self):
        """True only if the detector recently and explicitly said 0."""
        fresh = (time.time() - self.building_below_t) < BELOW_STALE
        return fresh and self.building_below == 0

    # ── drain altitude feedback socket to the newest value ──
    def _pump_altitude(self):
        while True:
            try:
                data, _ = self.alt_sock.recvfrom(256)
            except (BlockingIOError, OSError):
                break
            try:
                d = json.loads(data.decode())
                self.current_alt = float(d['alt'])
                if 'mission_alt' in d:
                    self.mission_alt = float(d['mission_alt'])
            except (ValueError, KeyError):
                continue

    # ── UDP command out: vz is up-positive (negative = descend),
    #    hold=True means stop forward motion ──
    def _send_cmd(self, vz_up, hold):
        msg = json.dumps({'vz': float(vz_up), 'hold': bool(hold)}).encode()
        try:
            self.cmd_sock.sendto(msg, self.cmd_addr)
        except OSError:
            pass

    # ── avoidance state machine ──
    def _run_avoidance(self):
        self._pump_altitude()
        tgt = self.target

        blocking    = tgt is not None and tgt['top_offset_px'] > 0
        in_far_band = blocking and OBSTACLE_NEAR <= tgt['depth'] < OBSTACLE_FAR
        too_close   = blocking and tgt['depth'] < OBSTACLE_NEAR
        cleared     = (tgt is None) or (tgt['top_offset_px'] < 0)

        above_mission = (
            self.current_alt is not None
            and self.mission_alt is not None
            and self.current_alt > self.mission_alt + ALT_TOL
        )

        if self.state == 'SEARCH':
            if too_close:
                self.get_logger().info(
                    f"<3m (depth={tgt['depth']:.1f}m) -> STOP + CLIMB ONLY")
                self._send_cmd(CLIMB_SPEED, True)      # hold + climb
                self.state = 'CLIMB'
            elif in_far_band:
                # climb+search band: drone1 keeps flying forward, we add vz
                self._send_cmd(CLIMB_SPEED, False)
            elif above_mission:
                # we are above cruise altitude: descend only over clear ground,
                # always keep moving forward (hold=False)
                if self._ground_clear():
                    self._send_cmd(-DESCEND_SPEED, False)
                else:
                    self._send_cmd(0.0, False)   # building below: hold alt, fly on
            else:
                self._send_cmd(0.0, False)       # at cruise, nothing ahead

        elif self.state == 'CLIMB':
            # locked into climb-only until the roofline clears
            self._send_cmd(CLIMB_SPEED, True)
            if cleared:
                # latch measured baseline; if no altitude yet, latch later
                # (later latch = MORE climb, never less)
                self.extra_start_alt = self.current_alt
                self.extra_start_time = time.time()
                if self.extra_start_alt is not None:
                    self.get_logger().info(
                        f"ROOFLINE CLEARED at {self.extra_start_alt:.2f} m -> "
                        f"climb to {self.extra_start_alt + EXTRA_CLIMB:.2f} m")
                else:
                    self.get_logger().warn(
                        "ROOFLINE CLEARED but no altitude feedback yet — "
                        "climbing until first altitude arrives")
                self.state = 'EXTRA_CLIMB'

        elif self.state == 'EXTRA_CLIMB':
            self._send_cmd(CLIMB_SPEED, True)

            # baseline not latched yet (no alt at clearing time)? latch now.
            if self.extra_start_alt is None and self.current_alt is not None:
                self.extra_start_alt = self.current_alt
                self.get_logger().info(
                    f"Altitude feedback online, baseline {self.extra_start_alt:.2f} m")

            done = (
                self.extra_start_alt is not None
                and self.current_alt is not None
                and self.current_alt >= self.extra_start_alt + EXTRA_CLIMB
            )

            timed_out = (time.time() - self.extra_start_time) > EXTRA_CLIMB_MAX_TIME
            if timed_out and not done:
                self.get_logger().warn(
                    "EXTRA_CLIMB safety timeout — altitude feedback missing? "
                    "Stopping climb (actual gain unknown but >> 6 m).")

            if done or timed_out:
                if done:
                    gained = self.current_alt - self.extra_start_alt
                    self.get_logger().info(
                        f"Extra climb complete (gain={gained:.2f} m) -> SEARCH "
                        f"(will descend to {self.mission_alt} m over clear ground)")
                else:
                    self.get_logger().info("Extra climb ended by timeout -> SEARCH")
                self.state = 'SEARCH'
                self.extra_start_alt = None
                self.extra_start_time = None

    def Controller(self):
        if self.DepthcamMessage is None:
            return
        depth_image = self.bridge.imgmsg_to_cv2(
            self.DepthcamMessage, desired_encoding='passthrough')
        depth_array_meters = np.array(depth_image, dtype=np.float32)  # * 0.001 only if 16UC1/mm
        depth_array_meters[depth_array_meters == 0] = 100.0
        h, w = depth_array_meters.shape
        center_row = h // 2
        center_col = w // 2

        half_h = 50
        half_w = 100
        r0, r1 = center_row - half_h, center_row + half_h
        c0, c1 = center_col - half_w, center_col + half_w
        roi = depth_array_meters[r0:r1, c0:c1]

        mask = (
            (roi >= self.range_min) &
            (roi <= self.range_max)
        ).astype(np.uint8)

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            mask, connectivity=8)

        obstacles = []
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area < 50:
                continue
            cx, cy = centroids[i]
            top_local = stats[i, cv2.CC_STAT_TOP]
            depth_val = float(np.median(roi[labels == i]))

            centroid_row = cy + r0
            top_row = top_local + r0

            obstacles.append({
                'depth': depth_val,
                'vertical_offset_px': center_row - centroid_row,
                'top_offset_px': center_row - top_row,
                'area': int(area),
            })

        if obstacles:
            self.target = min(obstacles, key=lambda o: o['depth'])
            t = self.target
            alt_str = f"{self.current_alt:.1f}m" if self.current_alt is not None else "n/a"
            print(f"[{self.state}] depth={t['depth']:.2f}m  "
                  f"top_off={t['top_offset_px']:.0f}px  alt={alt_str}  "
                  f"below={self.building_below}")
        else:
            self.target = None

        self._run_avoidance()


def main(args=None):
    rclpy.init(args=args)
    avoider_node = Avoider()
    rclpy.spin(avoider_node)
    avoider_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()