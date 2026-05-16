#!/usr/bin/env python3

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float32, String
from tf2_msgs.msg import TFMessage
from tf2_ros import Buffer, TransformException, TransformListener

try:
    from qcar2_interfaces.msg import MotorCommands
    HAS_INTERFACES = True
except ImportError:
    HAS_INTERFACES = False

try:
    from qcar2_object_detections.msg import (
        PersonDetection,
        StopSignDetection,
        TrafficLightDetection,
        ZebraCrossingDetection,
    )
    HAS_DETECTIONS = True
except ImportError:
    HAS_DETECTIONS = False

from .bt_nodes import (
    Action,
    BTNode,
    Condition,
    RepeatForever,
    Selector,
    Sequence,
    Status,
    TickContext,
    Wait,
)


@dataclass
class GoalPoint:
    x: float
    y: float
    qz: float  # quaternion z component
    qw: float  # quaternion w component


class QCar2BehaviorTreeManager(Node):
    def __init__(self):
        # type: () -> None
        super().__init__(
            'qcar2_behavior_tree_manager',
            automatically_declare_parameters_from_overrides=True,
        )

        try:
            self._declare_parameters()
        except Exception as e:
            self.get_logger().error(f'Error declaring parameters: {e}')
            raise

        try:
            self.goals = self._load_goals_from_parameters()
            self.get_logger().info(f'✓ Loaded {len(self.goals)} goals from parameters')
        except Exception as e:
            self.get_logger().error(f'Error loading goals: {e}')
            self.goals = []
        self.current_goal_index = -1
        self.current_goal = None  # type: Optional[GoalPoint]
        self.current_goal_start_sec = None  # type: Optional[float]
        self.goal_published_once = False
        self.mission_complete_announced = False

        self.person_detected = False
        self.stop_sign_detected = False
        self.zebra_detected = False
        self.traffic_light_detected = False
        self.traffic_light_state = 'unknown'
        self.mixer_state = 'unknown'
        self.last_motor_cmd = None

        self.tf_map_odom_ok = False
        self.tf_odom_base_ok = False
        self.tf_map_base_ok = False
        self.robot_x = 0.0
        self.robot_y = 0.0

        self.mode_hybrid = self.get_parameter('default_mode_hybrid').value

        self.goal_pub = self.create_publisher(
            PoseStamped,
            self.get_parameter('goal_output_topic').value,
            10,
        )
        self.state_pub = self.create_publisher(
            String,
            self.get_parameter('state_output_topic').value,
            10,
        )
        self.mode_pub = self.create_publisher(
            String,
            self.get_parameter('mode_output_topic').value,
            10,
        )
        self.mode_numeric_pub = self.create_publisher(
            Float32,
            self.get_parameter('mode_numeric_output_topic').value,
            10,
        )
        self.led_pub = self.create_publisher(
            String,
            self.get_parameter('led_output_topic').value,
            10,
        )

        if HAS_DETECTIONS:
            self.create_subscription(PersonDetection, '/detections/person', self._on_person, 10)
            self.create_subscription(StopSignDetection, '/detections/stop_sign', self._on_stop_sign, 10)
            self.create_subscription(TrafficLightDetection, '/detections/traffic_light', self._on_traffic_light, 10)
            self.create_subscription(ZebraCrossingDetection, '/detections/zebra_crossing', self._on_zebra, 10)
        else:
            self.get_logger().warn('qcar2_object_detections msgs not available - detection subs disabled')

        self.create_subscription(String, '/mixer/state', self._on_mixer_state, 10)

        if HAS_INTERFACES:
            self.create_subscription(MotorCommands, '/qcar2_motor_speed_cmd', self._on_motor_cmd, 10)
        else:
            self.get_logger().warn('qcar2_interfaces msgs not available - motor cmd sub disabled')

        self.create_subscription(TFMessage, '/tf', self._on_tf, 10)
        self.create_subscription(TFMessage, '/tf_static', self._on_tf_static, 10)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self, spin_thread=True)

        self.blackboard = {}  # type: Dict[str, Any]
        try:
            self.root = self._build_tree_from_parameters()
            self.get_logger().info('✓ BehaviorTree built successfully')
        except Exception as e:
            self.get_logger().error(f'Error building BehaviorTree: {e}')
            raise

        tick_hz = max(1.0, float(self.get_parameter('tick_hz').value))
        self.create_timer(1.0 / tick_hz, self._tick_tree)
        self.create_timer(0.1, self._update_tf_chain)

        self._publish_mode(self.mode_hybrid)
        self.get_logger().info(
            'BehaviorTree manager iniciado con {} goals | tick_hz={:.1f}'.format(
                len(self.goals), tick_hz
            )
        )

    def _declare_parameters(self):
        # type: () -> None
        self._declare_parameter_if_needed('tick_hz', 5.0)
        self._declare_parameter_if_needed('require_tf', True)
        self._declare_parameter_if_needed('goal_reached_distance', 0.35)
        self._declare_parameter_if_needed('goal_timeout_sec', 25.0)
        self._declare_parameter_if_needed('goal_frame_id', 'pgm_map')
        self._declare_parameter_if_needed('goal_output_topic', '/bt/goal')
        self._declare_parameter_if_needed('state_output_topic', '/bt/state')
        self._declare_parameter_if_needed('mode_output_topic', '/bt/mode_hybrid')
        self._declare_parameter_if_needed('mode_numeric_output_topic', '/bt/mode_hybrid_numeric')
        self._declare_parameter_if_needed('led_output_topic', '/btled')
        self._declare_parameter_if_needed('default_mode_hybrid', 'LANE_AND_NAV2')

        self._declare_parameter_if_needed('mode_code_stop', 0.0)
        self._declare_parameter_if_needed('mode_code_hybrid', 1.0)
        self._declare_parameter_if_needed('mode_code_pid', 2.0)

        self._declare_parameter_if_needed('goal_1', [1.0, 0.0, 0.0, 1.0])
        self._declare_parameter_if_needed('goal_2', [2.0, 0.5, 0.0, 1.0])
        self._declare_parameter_if_needed('goal_3', [3.0, 0.0, 0.0, 1.0])
        self._declare_parameter_if_needed('goal_4', [4.0, 0.0, 0.0, 1.0])
        # NOTE: additional_goals is NOT declared here because ROS 2 Humble
        # cannot infer the type of an empty list []. It is read via
        # try/except in _load_goals_from_parameters instead.

        default_tree = [
            'set_mode:AUTO_MISSION',
            'wait:1.0',
            'dispatch_next_goal',
            'wait_goal_reached_or_timeout',
            'wait:0.8',
        ]
        self._declare_parameter_if_needed('mission_loop', default_tree)

    def _declare_parameter_if_needed(self, name, default_value):
        # type: (str, Any) -> None
        if not self.has_parameter(name):
            self.declare_parameter(name, default_value)

    def _load_goals_from_parameters(self):
        # type: () -> List[GoalPoint]
        goals_raw = []  # type: List[Any]

        # ROS 2 Humble: No list_parameters() en Node.
        # Cargamos goal_1 a goal_100 secuencialmente hasta encontrar los que existen.
        for goal_num in range(1, 101):
            param_name = f'goal_{goal_num}'
            try:
                goal_value = self.get_parameter(param_name).value
                goals_raw.append(goal_value)
                self.get_logger().debug(f'  Loaded {param_name}: {goal_value}')
            except Exception:
                # Parámetro no existe, continuar
                continue

        # additional_goals is optional — may not be declared if YAML has []
        try:
            extra = self.get_parameter('additional_goals').value
        except Exception:
            extra = []
        if isinstance(extra, list):
            for point in extra:
                if isinstance(point, (list, tuple)) and len(point) >= 4:
                    goals_raw.append(point)

        goals = []  # type: List[GoalPoint]
        for point in goals_raw:
            if isinstance(point, (list, tuple)) and len(point) >= 4:
                goals.append(GoalPoint(
                    float(point[0]), 
                    float(point[1]), 
                    float(point[2]),  # qz
                    float(point[3])   # qw
                ))
        return goals

    # ── Detection callbacks ─────────────────────────────────────────────────

    def _on_person(self, msg):
        self.person_detected = bool(msg.detected)

    def _on_stop_sign(self, msg):
        self.stop_sign_detected = bool(msg.detected)

    def _on_traffic_light(self, msg):
        self.traffic_light_detected = bool(msg.detected)
        self.traffic_light_state = (msg.state or 'unknown').lower()

    def _on_zebra(self, msg):
        self.zebra_detected = bool(msg.detected)

    def _on_mixer_state(self, msg):
        self.mixer_state = msg.data

    def _on_motor_cmd(self, msg):
        self.last_motor_cmd = msg

    # ── TF callbacks ────────────────────────────────────────────────────────

    def _on_tf(self, msg):
        self._update_tf_flags_from_message(msg)

    def _on_tf_static(self, msg):
        self._update_tf_flags_from_message(msg)

    def _update_tf_flags_from_message(self, msg):
        frame_id = str(self.get_parameter('goal_frame_id').value).lstrip('/')
        for t in msg.transforms:
            parent = t.header.frame_id.lstrip('/')
            child = t.child_frame_id.lstrip('/')
            if parent == frame_id and child == 'odom':
                self.tf_map_odom_ok = True
            elif parent == 'odom' and child == 'base_link':
                self.tf_odom_base_ok = True

    def _update_tf_chain(self):
        frame_id = str(self.get_parameter('goal_frame_id').value).lstrip('/')

        map_to_odom = None
        map_to_base = None
        odom_to_base = None

        try:
            map_to_odom = self.tf_buffer.lookup_transform(frame_id, 'odom', rclpy.time.Time())
        except TransformException:
            map_to_odom = None

        try:
            odom_to_base = self.tf_buffer.lookup_transform('odom', 'base_link', rclpy.time.Time())
        except TransformException:
            odom_to_base = None

        try:
            map_to_base = self.tf_buffer.lookup_transform(frame_id, 'base_link', rclpy.time.Time())
        except TransformException:
            map_to_base = None

        # Fallback chain support when the tree is goal_frame -> map -> odom -> base_link
        if map_to_odom is None and frame_id != 'map':
            try:
                frame_to_map = self.tf_buffer.lookup_transform(frame_id, 'map', rclpy.time.Time())
                map_to_odom_fallback = self.tf_buffer.lookup_transform('map', 'odom', rclpy.time.Time())
                map_to_odom = map_to_odom_fallback if frame_to_map is not None else None
            except TransformException:
                map_to_odom = None

        self.tf_map_odom_ok = map_to_odom is not None
        self.tf_odom_base_ok = odom_to_base is not None
        self.tf_map_base_ok = map_to_base is not None

        if map_to_base is not None:
            self.robot_x = float(map_to_base.transform.translation.x)
            self.robot_y = float(map_to_base.transform.translation.y)

    # ── BT tree construction ────────────────────────────────────────────────

    def _build_tree_from_parameters(self):
        # type: () -> BTNode
        mission_step_strings = self.get_parameter('mission_loop').value
        if not isinstance(mission_step_strings, list):
            mission_step_strings = []

        mission_children = [
            Action('prepare_mission_state', self._action_publish_runtime_state),
            Condition('has_goals', lambda c: len(self.goals) > 0),
        ]  # type: List[BTNode]

        for idx, token in enumerate(mission_step_strings):
            mission_children.append(self._parse_step(str(token), idx))

        mission_children.append(Action('check_completed_mission', self._action_check_all_goals_done))
        mission_sequence = Sequence('mission_sequence', mission_children)

        operational = RepeatForever('operational_loop', mission_sequence)

        # Only add TF gate if require_tf is True
        require_tf = self.get_parameter('require_tf').value
        if require_tf:
            tf_gate = Selector(
                'tf_gate',
                [
                    Condition('tf_chain_ready', self._is_tf_chain_ready),
                    Action('wait_tf_state', self._action_tf_not_ready),
                ],
            )
            return Sequence('root', [tf_gate, operational])
        else:
            self.get_logger().warn('TF requirement DISABLED - goals will be published without localization')
            return operational

    def _parse_step(self, token, index):
        # type: (str, int) -> BTNode
        if token.startswith('wait:'):
            _, value = token.split(':', 1)
            return Wait('wait_{}'.format(index), float(value))

        if token.startswith('set_mode:'):
            _, mode = token.split(':', 1)
            return Action('set_mode_{}'.format(index), lambda c, m=mode: self._action_set_mode(c, m))

        if token.startswith('set_led:'):
            _, led_cmd = token.split(':', 1)
            return Action('set_led_{}'.format(index), lambda c, cmd=led_cmd: self._action_set_led(c, cmd))

        if token == 'dispatch_next_goal':
            return Action('dispatch_next_goal', self._action_dispatch_next_goal)

        if token == 'wait_goal_reached_or_timeout':
            return Action('wait_goal_reached_or_timeout', self._action_wait_goal_reached_or_timeout)

        return Action('noop_{}'.format(index), lambda c: Status.SUCCESS)

    # ── BT tick ─────────────────────────────────────────────────────────────

    def _tick_tree(self):
        try:
            self.blackboard['person_detected'] = self.person_detected
            self.blackboard['stop_sign_detected'] = self.stop_sign_detected
            self.blackboard['zebra_detected'] = self.zebra_detected
            self.blackboard['traffic_light_detected'] = self.traffic_light_detected
            self.blackboard['traffic_light_state'] = self.traffic_light_state
            self.blackboard['mixer_state'] = self.mixer_state
            self.blackboard['tf_ok'] = self._is_tf_chain_ready(None)
            self.blackboard['current_goal_index'] = self.current_goal_index

            context = TickContext(
                now_sec=self.get_clock().now().nanoseconds / 1e9,
                blackboard=self.blackboard,
                publish_state=self._publish_state,
            )

            status = self.root.tick(context)
            if status == Status.FAILURE:
                self._publish_state('BT_FAILURE')
                self.root.reset()
            elif status == Status.SUCCESS:
                self._publish_state('BT_SUCCESS')
                self.root.reset()
        except Exception as e:
            self.get_logger().error(f'Error in _tick_tree: {e}', exc_info=True)
            self._publish_state(f'BT_ERROR: {str(e)[:80]}')

    # ── BT action callbacks ────────────────────────────────────────────────

    def _is_tf_chain_ready(self, _):
        # type: (Any) -> bool
        return self.tf_map_odom_ok and self.tf_odom_base_ok and self.tf_map_base_ok

    def _action_tf_not_ready(self, _):
        # type: (TickContext) -> Status
        if self._is_tf_chain_ready(None):
            return Status.SUCCESS
        goal_frame = str(self.get_parameter('goal_frame_id').value).lstrip('/')
        self._publish_state(
            'WAITING_TF_CHAIN {}->odom->base_link (or {}->map->odom->base_link)'.format(
                goal_frame,
                goal_frame,
            )
        )
        return Status.RUNNING

    def _action_publish_runtime_state(self, _):
        # type: (TickContext) -> Status
        state = (
            'RUN idx={} '
            'person={} stop={} '
            'zebra={} tl={} mixer={}'
        ).format(
            self.current_goal_index,
            self.person_detected, self.stop_sign_detected,
            self.zebra_detected, self.traffic_light_state, self.mixer_state,
        )
        self._publish_state(state)
        return Status.SUCCESS

    def _action_set_mode(self, _, mode):
        # type: (TickContext, str) -> Status
        mode = self._normalize_mode_name(mode)
        if mode != self.mode_hybrid:
            self.mode_hybrid = mode
            self._publish_mode(mode)
        return Status.SUCCESS

    def _action_set_led(self, _, led_cmd):
        # type: (TickContext, str) -> Status
        """Publish LED command to /btled topic for led_sequence_node."""
        msg = String()
        msg.data = led_cmd.strip().lower()
        self.led_pub.publish(msg)
        self.get_logger().debug('LED command published: {}'.format(msg.data))
        return Status.SUCCESS

    def _action_dispatch_next_goal(self, _):
        # type: (TickContext) -> Status
        if self.current_goal is not None and not self._is_goal_terminal_state():
            return Status.SUCCESS

        next_index = self.current_goal_index + 1
        if next_index >= len(self.goals):
            if not self.mission_complete_announced:
                self._publish_state('MISSION_COMPLETE')
                self.mission_complete_announced = True
            return Status.SUCCESS

        self.current_goal_index = next_index
        self.current_goal = self.goals[self.current_goal_index]
        self.current_goal_start_sec = self.get_clock().now().nanoseconds / 1e9
        self.goal_published_once = False
        self.mission_complete_announced = False
        self._publish_current_goal()
        self._publish_state('GOAL_DISPATCHED index={}'.format(self.current_goal_index))
        return Status.SUCCESS

    def _action_wait_goal_reached_or_timeout(self, _):
        # type: (TickContext) -> Status
        if self.current_goal is None:
            return Status.SUCCESS

        if not self.goal_published_once:
            self._publish_current_goal()

        if self._goal_reached():
            self._publish_state('GOAL_REACHED index={}'.format(self.current_goal_index))
            self.current_goal = None
            self.current_goal_start_sec = None
            return Status.SUCCESS

        if self._goal_timeout():
            self._publish_state('GOAL_TIMEOUT index={} -> continue'.format(self.current_goal_index))
            self.current_goal = None
            self.current_goal_start_sec = None
            return Status.SUCCESS

        return Status.RUNNING

    def _action_check_all_goals_done(self, _):
        # type: (TickContext) -> Status
        if self.current_goal is None and (self.current_goal_index + 1) >= len(self.goals):
            self._publish_state('ALL_GOALS_DONE')
            return Status.SUCCESS
        return Status.SUCCESS

    # ── Goal utilities ──────────────────────────────────────────────────────

    def _goal_reached(self):
        # type: () -> bool
        if self.current_goal is None or not self.tf_map_base_ok:
            return False
        threshold = float(self.get_parameter('goal_reached_distance').value)
        distance = math.hypot(self.current_goal.x - self.robot_x, self.current_goal.y - self.robot_y)
        return distance <= threshold

    def _goal_timeout(self):
        # type: () -> bool
        if self.current_goal_start_sec is None:
            return False
        timeout = max(0.1, float(self.get_parameter('goal_timeout_sec').value))
        now_sec = self.get_clock().now().nanoseconds / 1e9
        return (now_sec - self.current_goal_start_sec) >= timeout

    def _is_goal_terminal_state(self):
        # type: () -> bool
        return False

    def _publish_current_goal(self):
        # type: () -> None
        if self.current_goal is None:
            return

        frame_id = self.get_parameter('goal_frame_id').value
        goal_msg = PoseStamped()
        goal_msg.header.stamp = self.get_clock().now().to_msg()
        goal_msg.header.frame_id = frame_id
        goal_msg.pose.position.x = self.current_goal.x
        goal_msg.pose.position.y = self.current_goal.y
        goal_msg.pose.position.z = 0.0

        # Use quaternion directly from goal parameters (qz, qw only)
        goal_msg.pose.orientation.x = 0.0
        goal_msg.pose.orientation.y = 0.0
        goal_msg.pose.orientation.z = self.current_goal.qz
        goal_msg.pose.orientation.w = self.current_goal.qw

        self.goal_pub.publish(goal_msg)
        self.goal_published_once = True
        
        # Debug: confirm quaternion is being published
        yaw = 2.0 * math.atan2(self.current_goal.qz, self.current_goal.qw)
        self.get_logger().info(
            f'📤 GOAL PUBLISHED index={self.current_goal_index}: '
            f'pos=({self.current_goal.x:.2f}, {self.current_goal.y:.2f}) '
            f'quat=(0, 0, {self.current_goal.qz:.3f}, {self.current_goal.qw:.3f}) '
            f'yaw={yaw:.3f}rad ({math.degrees(yaw):.1f}°)'
        )

    @staticmethod
    def _quaternion_from_yaw(yaw):
        # type: (float) -> Tuple[float, float, float, float]
        half = yaw * 0.5
        return (0.0, 0.0, math.sin(half), math.cos(half))

    # ── State & mode publishing ─────────────────────────────────────────────

    def _publish_state(self, text):
        # type: (str) -> None
        msg = String()
        msg.data = text
        self.state_pub.publish(msg)

    def _publish_mode(self, mode):
        # type: (str) -> None
        msg = String()
        msg.data = mode
        self.mode_pub.publish(msg)

        numeric_mode = Float32()
        numeric_mode.data = self._mode_to_numeric_code(mode)
        self.mode_numeric_pub.publish(numeric_mode)

    @staticmethod
    def _normalize_mode_name(raw_mode):
        # type: (str) -> str
        cleaned = (raw_mode or '').strip().upper()
        alias_map = {
            'AUTO_MISSION': 'HYBRID',
            'HYBRID': 'HYBRID',
            'LANE_AND_NAV2': 'HYBRID',
            'LANE_PID_ONLY': 'LANE_PID',
            'LANE_PID': 'LANE_PID',
            'LANE_ONLY': 'LANE_ONLY',
            'NAV2_TURN': 'NAV2_TURN',
            'NAV2_FORCED': 'NAV2_FORCED',
            'STOPPED': 'STOPPED',
        }
        return alias_map.get(cleaned, 'HYBRID')

    def _mode_to_numeric_code(self, mode):
        # type: (str) -> float
        normalized = self._normalize_mode_name(mode)
        mode_map = {
            'STOPPED': float(self.get_parameter('mode_code_stop').value),
            'HYBRID': float(self.get_parameter('mode_code_hybrid').value),
            'NAV2_TURN': float(self.get_parameter('mode_code_hybrid').value),
            'NAV2_FORCED': float(self.get_parameter('mode_code_hybrid').value),
            'LANE_PID': float(self.get_parameter('mode_code_pid').value),
            'LANE_ONLY': float(self.get_parameter('mode_code_pid').value),
        }
        return mode_map.get(normalized, float(self.get_parameter('mode_code_hybrid').value))


def main(args=None):
    # type: (Optional[List[str]]) -> None
    rclpy.init(args=args)
    node = QCar2BehaviorTreeManager()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
