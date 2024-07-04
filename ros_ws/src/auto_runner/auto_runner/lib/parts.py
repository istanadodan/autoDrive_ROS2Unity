from enum import Enum, auto
from auto_runner.lib.common import (
    EvHandle,
    StateData,
    MessageHandler,
    State,
    Orient,
    DirType,
    Observable,
    Message,
    Observer,
    Chainable,
    stop_event,
)
from auto_runner.lib.path_manage import PathManage
import math
import time
from auto_runner import mmr_sampling

state: StateData = StateData(State.ROTATE_STOP, State.ROTATE_STOP)
orient: StateData = StateData(Orient.X, Orient.X)

print_log = mmr_sampling.print_log


class PolicyType(Enum):
    ROTATE = "rotate"
    STRAIGHT = "move"
    BACk = "go_back"


class MapData(Observable):
    _subject = "map"


class LidarData(Observable):
    _subject = "lidar"


class IMUData(Observable):
    _subject = "imu"

    @classmethod
    def publish_(cls, data: object):
        _data = [
            data.twist.linear.x,
            data.twist.linear.y,
            data.twist.angular.z,
        ]
        super().publish_(_data)


class MoveAction(EvHandle, Observer, Chainable):

    def __init__(self, orient: Orient, cur_pos: tuple, path: list[tuple]):
        super().__init__()
        self.orient = orient
        self.cur_pos = cur_pos
        self.path = path
        self.integral_error = 0
        self.previous_error = None

    def setup(self, **kwargs):
        self.imu_data = None
        self.next_pos = None
        IMUData.subscribe(o=self)

    def check_condition(self):
        path = self.path
        if not path:
            return False

        orient = self.orient
        cur_pos = self.cur_pos
        (x0, y0), (x1, y1) = cur_pos, path[1]
        output: bool = False

        if orient in [Orient.X, Orient._X]:
            output = x0 != x1
        elif orient in [Orient.Y, Orient._Y]:
            output = y0 != y1

        if output == True:
            self.action = DirType.FORWARD, orient
            self.torque = 0.5

            ix = len(path) - 1
            for index, pos in enumerate(path[1:], start=1):
                if orient in [Orient.X, Orient._X]:
                    if pos[1] != y0:
                        ix = index - 1
                        break
                elif orient in [Orient.Y, Orient._Y]:
                    if pos[0] != x0:
                        ix = index - 1
                        break
            # 회전 직적까지 후진
            self.next_pos = path[ix]

        return output

    def run(self):
        if state != State.ROTATE_STOP:
            # 회전중이라면 회전을 중지시킨다.
            stop_event.set()
            time.sleep(0.3)
            stop_event.clear()

        state.shift(State.RUN)
        while not stop_event.is_set():
            imu_data = self.get_msg().data

            with self._lock:
                self.cur_pos = PathManage.transfer2_xy(imu_data[:2])
                print_log(f"cur: {self.cur_pos} : dest: {self.next_pos}")

                # 도착 여부 체크
                if self.cur_pos == self.next_pos:
                    break

                _angle = imu_data[-1]
                _next_orient = self.action[1]

                amend_theta: float = self._adjust_body(_next_orient, _angle)

                self._notifyall(
                    "node",
                    Message(
                        title="move",
                        data_type="command",
                        data=dict(
                            torque=self._pid_torque(imu_data[:2]), theta=amend_theta
                        ),
                    ),
                )

        self._notifyall(
            "node", Message(title="공지", data_type="notify", data="Move 완료")
        )

        state.shift(State.STOP)
        IMUData.unsubscribe(o=self)

    # PID제어를통한 torque 계산
    def _pid_torque(self, cur_pos: tuple[float, float]) -> float:
        # 에러정의
        Kp, Ki, Kd, dt = 0.3, 0.03, 0.6, 3.0
        max_integral = 1.0
        next_pos = PathManage.transfer2_point(self.next_pos)

        error = math.sqrt(
            (next_pos[0] - cur_pos[0]) ** 2 + (next_pos[1] - cur_pos[1]) ** 2
        )

        self.integral_error = min(self.integral_error + error, max_integral) / dt

        derivative_error = (
            error - (self.previous_error if self.previous_error else 0.0)
        ) / dt

        self.previous_error = error

        if error <= 1.0:
            self.integral_error = 0  # 오차가 매우 작을 때 적분항 리셋

        output = Kp * error + Ki * self.integral_error + Kd * derivative_error

        print_log(
            f"dynamic torque = {output}: {error=}, {self.integral_error=}, {derivative_error=}"
        )
        return output

    # 수평상태
    def _adjust_body(self, orient: Orient, angle: float):
        amend_theta: float = self._get_deviation_radian(orient, angle)

        # 간격을 두어 보정한다.
        # 잦은 조정에의한 좌우 흔들림 방지.
        if abs(amend_theta) > 3 / 180 * math.pi:
            # print_log(
            #     f"[{self.dir_data.cur} 위치보정] {self.dir_data} >> 보정 각:{amend_theta} / 기준:{3 / 180 * math.pi}"
            # )
            pass
        else:
            amend_theta = 0.0

        return amend_theta * 0.8

    # 각도를 입력받아, 진행방향과 벗어난 각도를 반환한다.
    def _get_deviation_radian(self, orient: Orient, angle: float) -> float:
        # 유니티에서 로봇의 기본회전상태가 180 CCW상태, 즉 음의 값.
        # 회전방향 수치를 양수로 정리한다.
        # -0.1 => 6.23
        _angle = angle % (math.pi * 2)

        if orient == Orient.X:
            amend_theta = -_angle if _angle < math.pi / 2 else 2 * math.pi - _angle

        elif orient == Orient._X:
            amend_theta = math.pi - _angle

        elif orient == Orient.Y:
            # -180 ~ -270
            # 차이량에 대해 음수면 바퀴가 우측방향으로 돌게된다. (우회전으로 보정)
            amend_theta = math.pi / 2 - _angle

        elif orient == Orient._Y:
            amend_theta = math.pi * 3 / 2 - _angle

        else:
            amend_theta = 0.0

        return amend_theta


class RotateAction(EvHandle, Observer, Chainable):
    start_angle: float

    rotate_map = {"basic": (0.3, 0.15), "fast": (0.4, 0.20)}

    def __init__(self, orient: Orient, cur_pos: tuple, path: list[tuple]):
        super().__init__()
        self.orient = orient
        self.cur_pos = cur_pos
        self.path = path
        self.rotate_policy = self.rotate_map["fast"]

    def setup(self, **kwargs):
        self.imu_data: list = None
        IMUData.subscribe(o=self)

    def check_condition(self):
        path = self.path
        if not path:
            return False

        orient = self.orient
        cur_pos = self.cur_pos
        (x0, y0), (x1, y1) = cur_pos, path[1]

        action = None
        if orient == Orient.X:
            if y1 > y0:
                action = DirType.LEFT, Orient.Y  # 좌회전
            elif y1 < y0:
                action = DirType.RIGHT, Orient._Y  # 우회전

        elif orient == Orient._X:
            if y1 > y0:
                action = DirType.RIGHT, Orient.Y  # 우회전
            elif y1 < y0:
                action = DirType.LEFT, Orient._Y  # 좌회전

        elif orient == Orient.Y:
            if x1 > x0:
                action = DirType.RIGHT, Orient.X  # 우회전
            elif x1 < x0:
                action = DirType.LEFT, Orient._X  # 좌회전

        elif orient == Orient._Y:
            if x1 > x0:
                action = DirType.LEFT, Orient.X  # 좌회전
            elif x1 < x0:
                action = DirType.RIGHT, Orient._X  # 우회전

        if action is None:
            return False

        self.action = action
        return True

    def run(self):
        if state == State.ROTATE_START:
            stop_event.set()
            time.sleep(0.3)
            stop_event.clear()

        state.shift(State.ROTATE_START)

        while not stop_event.is_set():
            imu_data = self.get_msg().data

            with self._lock:
                _fr_angle = imu_data[-1]
                _to_angle = self._get_target_angle()

                # 회전각 계산
                angle_diff: float = self._get_diff(_fr_angle, _to_angle)

                # 회전완료 여부 체크
                if abs(angle_diff) < self.rotate_policy[0]:
                    break

                sign_ = 1 if self.action[0] == DirType.LEFT else -1
                self._notifyall(
                    "node",
                    Message(
                        data_type="command",
                        data=dict(
                            name="회전",
                            torque=self.rotate_policy[1],
                            theta=sign_ * 1.35,
                        ),
                    ),
                )

        self._notifyall("node", Message(data_type="notify", data=dict(name="rotate")))

        state.shift(State.ROTATE_STOP)
        IMUData.unsubscribe(o=self)

    # 목표 각도 설정
    def _get_target_angle(self) -> float:
        _angle_map = {
            Orient.X: 2 * math.pi if self.action[0] == DirType.LEFT else 0,
            Orient.Y: math.pi / 2,
            Orient._X: math.pi,
            Orient._Y: 3 * math.pi / 2,
        }

        _key = self.action[1]  # 다음 방향
        return _angle_map[_key] if _key in _angle_map else 2 * math.pi

    # 반환값은 보정될 방향을 고려하여 계산방향을 정함
    def _get_diff(self, angle: float, target_angle: float) -> float:
        cur_angle = angle % (2 * math.pi)
        # cur_angle = cur_angle if cur_angle > target_angle else (cur_angle + 2 * math.pi)

        print_log(f"rotate: {cur_angle} : {target_angle}")
        return target_angle - cur_angle


class BackMoveAction(EvHandle, Observer, Chainable):
    def __init__(self, orient: Orient, cur_pos: tuple, path: list[tuple]):
        super().__init__()
        self.orient = orient
        self.cur_pos = cur_pos
        self.path = path

    def setup(self, **kwargs):
        self.integral_error = 0
        self.previous_error = None
        self.next_pos = None
        self.imu_data = None
        IMUData.subscribe(o=self)

    def _is_backward(self, cur_pos: tuple, next_pos: tuple, orient: Orient) -> bool:
        x0, y0 = cur_pos
        x1, y1 = next_pos
        if orient == Orient.X:
            return x0 > x1
        elif orient == Orient._X:
            return x0 < x1
        elif orient == Orient.Y:
            return y0 > y1
        elif orient == Orient._Y:
            return y0 < y1
        else:
            raise Exception("Wrong orient entered")

    def check_condition(self):
        path = self.path
        orient = self.orient
        cur_pos = self.cur_pos
        next_pos = path[1]

        output = self._is_backward(cur_pos, next_pos, orient)
        if output == True:
            self.action = DirType.BACKWARD, orient
            x0, y0 = cur_pos
            ix = len(path) - 1
            for index, pos in enumerate(path[1:], start=1):
                if orient in [Orient.X, Orient._X]:
                    if pos[1] != y0:
                        ix = index - 1
                        break
                elif orient in [Orient.Y, Orient._Y]:
                    if pos[0] != x0:
                        ix = index - 1
                        break
            # 회전 직적까지 후진
            self.next_pos = path[ix]

        return output

    def run(self):
        if state != State.ROTATE_STOP:
            # 회전중이라면 회전을 중지시킨다.
            self.stop_event.set()
            time.sleep(0.3)
            self.stop_event.clear()

        state.shift(State.RUN)
        while not self.stop_event.is_set():
            imu_data = self.get_msg().data

            with self._lock:
                _xy = imu_data[:2]
                # 도착 여부 체크
                _dir = self.action[0]
                if PathManage.transfer2_xy(_xy, _dir) == self.next_pos:
                    break

                _angle = self.imu_data[-1]
                _next_orient = self.action[1]
                amend_theta: float = self._adjust_body(_next_orient, _angle)

                self._notifyall(
                    "node",
                    Message(
                        title="후진",
                        data_type="command",
                        data=dict(
                            torque=-self._pid_torque(imu_data[:2]), theta=-amend_theta
                        ),
                    ),
                )

        self._notifyall("node", Message(title="완료", data_type="notify", data="move"))

        state.shift(State.ROTATE_STOP)
        IMUData.unsubscribe(o=self)

    # x,y,angle.z
    def update(self, message: Message):
        match message.data_type:
            case "imu":
                self.imu_data = message.data
            case "map":
                self.map_data = message.data

    # 수평상태
    def _adjust_body(self, orient: Orient, angle: float):
        amend_theta: float = self._get_deviation_radian(orient, angle)

        # 간격을 두어 보정한다.
        # 잦은 조정에의한 좌우 흔들림 방지.
        if abs(amend_theta) > 3 / 180 * math.pi:
            # print_log(
            #     f"[{self.dir_data.cur} 위치보정] {self.dir_data} >> 보정 각:{amend_theta} / 기준:{3 / 180 * math.pi}"
            # )
            pass
        else:
            amend_theta = 0

        return amend_theta

    # 각도를 입력받아, 진행방향과 벗어난 각도를 반환한다.
    def _get_deviation_radian(self, orient: Orient, angle: float) -> float:
        # 유니티에서 로봇의 기본회전상태가 180 CCW상태, 즉 음의 값.
        # 회전방향 수치를 양수로 정리한다.
        # -0.1 => 6.23
        _angle = angle % (math.pi * 2)

        if orient == Orient.X:
            amend_theta = -_angle if _angle < math.pi / 2 else 2 * math.pi - _angle

        elif orient == Orient._X:
            amend_theta = math.pi - _angle

        elif orient == Orient.Y:
            # -180 ~ -270
            # 차이량에 대해 음수면 바퀴가 우측방향으로 돌게된다. (우회전으로 보정)
            amend_theta = math.pi / 2 - _angle

        elif orient == Orient._Y:
            amend_theta = math.pi * 3 / 2 - _angle

        else:
            amend_theta = 0.0

        return amend_theta

    # PID제어를통한 torque 계산
    def _pid_torque(self, cur_pos: tuple[float, float]) -> float:
        # 에러정의
        Kp, Ki, Kd, dt = 0.3, 0.03, 0.6, 3.0
        max_integral = 1.0
        next_pos = PathManage.transfer2_point(self.next_pos)

        error = math.sqrt(
            (next_pos[0] - cur_pos[0]) ** 2 + (next_pos[1] - cur_pos[1]) ** 2
        )

        self.integral_error = min(self.integral_error + error, max_integral) / dt

        derivative_error = (
            error - (self.previous_error if self.previous_error else 0.0)
        ) / dt

        self.previous_error = error

        if error <= 1.0:
            self.integral_error = 0  # 오차가 매우 작을 때 적분항 리셋

        output = Kp * error + Ki * self.integral_error + Kd * derivative_error

        print_log(
            f"dynamic torque = {output}: {error=}, {self.integral_error=}, {derivative_error=}"
        )
        return output


# 다음 처리방향을 세운다.
class Policy:

    def setup(self, policy: PolicyType, **kwargs):
        if not hasattr(self, policy.value):
            raise Exception(f"Policy {policy} is not implemented.")
        # factory 패턴
        self.action: EvHandle = getattr(self, policy.value)(**kwargs)

    @classmethod
    def search_action(
        cls, orient: Orient, cur_pos: tuple, path: list[tuple]
    ) -> EvHandle:
        print_log(f"search_action: {orient}, {cur_pos}, {path}")
        # 체인 패턴
        for p in [
            RotateAction,
            MoveAction,
            BackMoveAction,
        ]:
            c: Chainable = p(orient=orient, cur_pos=cur_pos, path=path)
            if c.check_condition():
                return c


class ObstacleManager:
    def __init__(self, node: MessageHandler, state_data: StateData):
        self.node = node
        self.state_data = state_data

    # 회전과 이동이 구독
    def subscribe(self, o: object):
        self.observers.append(o)
