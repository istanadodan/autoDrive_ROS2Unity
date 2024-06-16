import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from rclpy.time import Time
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup
from geometry_msgs.msg import TwistStamped, Twist, Vector3
from yolov8_msgs.srv import CmdMsg
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import OccupancyGrid
from auto_runner.map_transform import convert_map
from auto_runner.lib import car_drive, common, path_location

laser_scan: LaserScan = None
grid_map: list[list[int]] = None


class GridMap(Node):
    def __init__(self) -> None:
        super().__init__("grid_map_node")

        # 기본 콜백 그룹 생성
        self._default_callback_group = ReentrantCallbackGroup()

        # 맵 수신 10*10
        self.map_pub = self.create_subscription(
            OccupancyGrid,
            "/occ_grid_map",
            self.receive_map,
            10,
        )

    def receive_map(self, map: OccupancyGrid):
        global grid_map
        raw_data = np.array(map.data, dtype=np.int8)
        grid_map = convert_map(raw_data)


class LidarScanNode(Node):
    def __init__(self) -> None:
        super().__init__("lidar_scan_node")

        # 기본 콜백 그룹 생성
        self._default_callback_group = ReentrantCallbackGroup()

        # lidar subscriber
        self.create_subscription(
            LaserScan,
            "jetauto/scan",
            callback=self.scan_handler,
            qos_profile=10,
        )

        self.get_logger().info(f"LidarScanNode started...")

    def scan_handler(self, msg: LaserScan):
        global laser_scan
        laser_scan = msg


class AStartSearchNode(Node):

    def __init__(self) -> None:
        super().__init__("path_search_astar")
        queue_size = 1
        self.twist_msg = Twist()

        # odom 위치정보 취득
        qos_profile = QoSProfile(depth=queue_size)
        self.create_subscription(
            TwistStamped,
            "/unity_tf",
            callback=self.unity_tf_callback,
            qos_profile=qos_profile,
        )
        self.pub_jetauto_car = self.create_publisher(
            Twist, "jetauto_car/cmd_vel", queue_size
        )

        self.srv_jetauto_car = self.create_service(
            CmdMsg, "jetauto_car/cmd_msg", self.send_command
        )

        self.robot_ctrl = car_drive.RobotController(node=self, dir=common.Orient.X)
        
        self.setup()

        self.get_logger().info(f"AStart_Path_Search_Mode has started...")

    def unity_tf_callback(
        self, loc_data: TwistStamped
    ) -> None:  # msg로부터 위치정보를 추출
        if not grid_map:
            return

        # 맵,위치데이터 수신
        self.robot_ctrl.update_map(grid_map)
        self.robot_ctrl.update_pos(loc_data)

        if self.robot_ctrl.check_arrival(finish=False):
            return

        is_near = self._is_near()
        if is_near:
            # 급 감속
            action = self.state_near
            self._send_message(title="근접 후진", x=action[0], theta=action[1])
            return

        if self.robot_ctrl.check_rotate_state():
            return

        # 방향 보정
        if self.robot_ctrl.adjust_body():
            return

        # 로봇 다음동작
        try:
            x, theta = self.robot_ctrl.next_action()
        except Exception as e:
            self.get_logger().info(f"controller stopped:{e}")
            # 맵 생성까지 대기
            return

        self.robot_ctrl.break_if_rotating()

        # 제어메시지 발행
        self._send_message(title="주행지시", x=x, theta=theta)

    # 로봇이 전방물체와 50cm이내 접근상태이면 True를 반환
    def _is_near(self) -> tuple[bool, tuple]:
        if laser_scan:
            time = Time.from_msg(laser_scan.header.stamp)
            # Run if only laser scan from simulation is updated
            if time.nanoseconds > self.nanoseconds:
                self.nanoseconds = time.nanoseconds

                ### Driving ###
                nearest_distance = 0.3  # 50cm

                distance_map = {}
                distance_map.update({0: min(laser_scan.ranges[0:8])})  # 우측
                distance_map.update({1: min(laser_scan.ranges[8:16])})
                distance_map.update({2: min(laser_scan.ranges[16:24])})
                distance_map.update({3: min(laser_scan.ranges[24:28])})

                distance_map.update({4: min(laser_scan.ranges[28:37])})  # 중앙

                distance_map.update({5: min(laser_scan.ranges[37:41])})  # 좌측
                distance_map.update({6: min(laser_scan.ranges[41:49])})
                distance_map.update({7: min(laser_scan.ranges[49:57])})
                distance_map.update({8: min(laser_scan.ranges[57:65])})

                min_distance = distance_map.get(4)
                # map에서 value로 index를 찾는다.
                torq_map = {
                    2: -0.2,
                    3: -0.5,
                    4: -0.6,
                    5: -0.5,
                    6: -0.2,
                }
                min_index = next(
                    key for key, value in distance_map.items() if value == min_distance
                )
                angle = -1.0 if min_index < 4 else 0.0 if min_index == 4 else 1.0
                torque = torq_map.get(min_index, 0.0)

                self.state_near = (torque, angle)

                self.get_logger().info(
                    f"distances:{min_distance}/기준:{nearest_distance}"
                )
                self.get_logger().info(f"index:{min_index}, T/A: {torque}/{angle}")
                return min_distance <= nearest_distance

        return False

    def setup(self) -> None:
        self.nanoseconds = 0
        self.get_logger().info("초기화 처리 완료")

    def send_command(self, request, response) -> None:
        self.get_logger().info(f"request: {request}")
        response.success = True
        if request.message == "reset":
            self.setup()
        elif request.message == "set_dest":
            x = int(request.data.x)
            y = int(request.data.y)
            self.set_destpos((x, y))
        elif request.message == "set_torque":
            self.FWD_TORQUE = request.data.x
        else:
            response.success = False

        return response

    def _send_message(
        self, *, title: str = None, x: float = 0.0, theta: float = 0.0
    ) -> None:
        self.get_logger().info(f"#### {title} ### torque:{x}, angular:{theta} ####")
        self.twist_msg.linear = Vector3(x=x, y=0.0, z=0.0)
        self.twist_msg.angular = Vector3(x=0.0, y=0.0, z=theta)
        self.pub_jetauto_car.publish(self.twist_msg)

    def print_log(self, message) -> None:
        self.get_logger().info(message)


def main(args=None):
    rclpy.init(args=args)

    path_node = AStartSearchNode()
    lidar_node = LidarScanNode()
    grid_node = GridMap()

    executor = MultiThreadedExecutor()
    executor.add_node(grid_node)
    executor.add_node(path_node)
    executor.add_node(lidar_node)

    try:
        executor.spin()
    finally:
        executor.shutdown()
        path_node.destroy_node()
        rclpy.shutdown()
