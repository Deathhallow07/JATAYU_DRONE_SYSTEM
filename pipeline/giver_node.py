
import threading
import math

import rclpy
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor
from sensor_msgs.msg import Image, NavSatFix, Imu
from std_msgs.msg import Float64
import message_filters
from cv_bridge import CvBridge
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from scipy.spatial.transform import Rotation


class Giver(Node):
    

    TELEMETRY_FIELDS = [
        'timestamp_sec',  
        'longitude',
        'altitude',       
        'roll',
        'pitch',
        'yaw',            
        'compass_deg',    
    ]

    def __init__(self, node_name: str = 'giver_node', max_cache_age: float = 1.0):
        if not rclpy.ok():
            rclpy.init(args=None)

        super().__init__(node_name)

        self._altitude_cache = None
        self._compass_cache = None
        self.max_cache_age = max_cache_age
        self.bridge = CvBridge()

        self._lock = threading.Lock()
        self._latest_pair = None

        mavros_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=20
        )
        reliable_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=20
        )

        self.create_subscription(Float64, '/mavros/global_position/rel_alt',
                                  self._altitude_cb, mavros_qos)
        self.create_subscription(Float64, '/mavros/global_position/compass_hdg',
                                  self._compass_cb, mavros_qos)

        sub_image = message_filters.Subscriber(
            self, Image, '/camera/camera/color/image_raw', qos_profile=reliable_qos)
        sub_gps = message_filters.Subscriber(
            self, NavSatFix, '/mavros/global_position/global', qos_profile=mavros_qos)
        sub_imu = message_filters.Subscriber(
            self, Imu, '/mavros/imu/data', qos_profile=mavros_qos)

        self.sync = message_filters.ApproximateTimeSynchronizer(
            [sub_image, sub_gps, sub_imu],
            queue_size=20,
            slop=0.1,
            allow_headerless=False
        )
        self.sync.registerCallback(self._synced_callback)

        # --- spin this node on its own background thread ---
        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self)
        self._spin_thread = threading.Thread(target=self._spin, daemon=True)
        self._stop_event = threading.Event()
        self._spin_thread.start()

        self.get_logger().info('Giver started, spinning in background thread')


    def _altitude_cb(self, msg: Float64):
        now = self.get_clock().now().nanoseconds * 1e-9
        self._altitude_cache = (msg.data, now)

    def _compass_cb(self, msg: Float64):
        now = self.get_clock().now().nanoseconds * 1e-9
        self._compass_cache = (msg.data, now)

    def _get_cached(self, cache, name: str):
        if cache is None:
            self.get_logger().warn(f'No {name} data received yet',
                                    throttle_duration_sec=5.0)
            return None
        value, ts = cache
        age = self.get_clock().now().nanoseconds * 1e-9 - ts
        if age > self.max_cache_age:
            self.get_logger().warn(
                f'{name} cache is {age:.2f}s old (max {self.max_cache_age}s) — skipping frame',
                throttle_duration_sec=2.0
            )
            return None
        return value

    @staticmethod
    def _imu_to_euler(imu_msg: Imu):
        q = imu_msg.orientation
        yaw, pitch, roll = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_euler('ZYX')
        return roll, pitch, yaw

    def _synced_callback(self, img_msg: Image, gps_msg: NavSatFix, imu_msg: Imu):
        altitude = self._get_cached(self._altitude_cache, 'altitude')
        if altitude is None:
            return

        compass = self._get_cached(self._compass_cache, 'compass')
        if compass is None:
            return

        drone_lat = gps_msg.latitude
        drone_lon = gps_msg.longitude
        roll, pitch, _yaw_imu = self._imu_to_euler(imu_msg)
        yaw = math.radians(compass)  # compass heading takes priority over IMU yaw
        stamp = img_msg.header.stamp
        timestamp_sec = stamp.sec + stamp.nanosec * 1e-9

        try:
            cv_image = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'cv_bridge error: {e}')
            return

        telemetry = {
            'timestamp_sec': timestamp_sec,
            'latitude': drone_lat,
            'longitude': drone_lon,
            'altitude': altitude,
            'roll': roll,
            'pitch': pitch,
            'yaw': yaw,
            'compass_deg': compass,
        }


        with self._lock:
            self._latest_pair = (telemetry, cv_image.copy())


    def get_latest(self):
        
        with self._lock:
            return self._latest_pair

    def read(self):
        
        pair = self.get_latest()
        if pair is None:
            return False, None, {}
        telemetry, frame = pair
        return True, frame, telemetry

    def _spin(self):
        try:
            self._executor.spin()
        except Exception:
            pass

    def shutdown(self):
        
        try:
            self._executor.shutdown()
        except Exception:
            pass
        self.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()



def main(args=None):
    giver = Giver()
    try:
        giver._spin_thread.join()
    except KeyboardInterrupt:
        pass
    finally:
        giver.shutdown()


if __name__ == '__main__':
    main()