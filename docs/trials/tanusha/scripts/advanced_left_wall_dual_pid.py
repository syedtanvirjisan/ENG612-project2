
"""ENG612 Project 2: TurtleBot3 Burger extended dual-PID wall follower.

Angular PID tracks 0.30 m from the LEFT wall.
Linear PID tracks a scheduled forward-speed reference using /odom.
Tune PID gains and speed limits live via ROS 2 parameters.
Test in Gazebo before physical deployment.
"""

import csv
import math
import time
from datetime import datetime
from pathlib import Path

import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rcl_interfaces.msg import SetParametersResult
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan


def clamp(value, low, high):
    return max(low, min(high, value))


def angle_difference(target, current):
    return math.atan2(
        math.sin(target - current),
        math.cos(target - current)
    )


class ScanReaderLeft(Node):

    def __init__(self):
        super().__init__('scan_reader_left')

        # Navigation thresholds are fixed for controlled comparisons.
        self.target = 0.30
        self.max_left = 0.45
        self.min_left = 0.25
        self.front_limit = 0.35
        self.front_rearm = 0.45
        self.front_left_limit = 0.22
        self.max_turn_speed = 0.8
        self.recovery_turn_speed = 0.3
        self.wall_slowdown_error = 0.15
        self.front_slowdown_start = 0.65

        # Live-adjustable parameters: 6 gains plus 3 speed settings.
        self.defaults = {
            'kp': 1.0,
            'ki': 0.01,
            'kd': 0.50,
            'kp_linear': 0.30,
            'ki_linear': 0.04,
            'kd_linear': 0.02,
            'minimum_speed': 0.07,
            'nominal_speed': 0.09,
            'maximum_speed': 0.1,
        }

        for name, default in self.defaults.items():
            self.declare_parameter(name, default)

        settings = {
            name: self.get_parameter(name).value
            for name in self.defaults
        }

        reason = self.validate_settings(settings)
        if reason:
            raise ValueError(reason)

        for name, value in settings.items():
            setattr(self, name, float(value))

        self.yaw = None
        self.measured_speed = None
        self.last_scan = None
        self.last_odom = None
        self.state = 'FOLLOW'
        self.state_since = time.monotonic()
        self.turn_target = None
        self.corner_armed = True

        self.reset_pid()

        folder = Path.home() / 'wall_follower_logs'
        folder.mkdir(parents=True, exist_ok=True)

        run_name = datetime.now().strftime(
            'left_pid_run_%Y%m%d_%H%M%S_%f'
        )

        self.csv_path = folder / f'{run_name}.csv'
        self.csv_file = self.csv_path.open(
            'w', newline='', encoding='utf-8'
        )

        self.writer = csv.writer(self.csv_file)

        self.writer.writerow([
            'scan_time_s',
            'front_m',
            'left_m',
            'desired_m',
            'error_m',
            'linear_m_s',
            'angular_rad_s',
            'state',
            'front_left_m',
            'yaw_deg',
            'speed_reference_m_s',
            'measured_speed_m_s',
            'speed_error_m_s',
            'kp',
            'ki',
            'kd',
            'kp_linear',
            'ki_linear',
            'kd_linear',
            'minimum_speed',
            'nominal_speed',
            'maximum_speed',
        ])

        (folder / f'{run_name}.settings.txt').write_text(
            '\n'.join(
                f'{name}={getattr(self, name)}'
                for name in self.defaults
            ) + '\n',
            encoding='utf-8'
        )

        self.change_log = (
            folder / f'{run_name}.changes.csv'
        ).open('w', newline='', encoding='utf-8')

        self.change_writer = csv.writer(self.change_log)

        self.change_writer.writerow([
            'wall_time_s', 'parameter', 'new_value'
        ])

        self.change_log.flush()

        self.pub = self.create_publisher(
            Twist, '/cmd_vel', 10
        )

        self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            qos_profile_sensor_data
        )

        self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            qos_profile_sensor_data
        )

        self.create_timer(0.1, self.watchdog)
        self.add_on_set_parameters_callback(
            self.update_parameters
        )

        self.get_logger().info(
            'LEFT WALL: angular PID + linear speed PID'
        )

        self.get_logger().info(
            f'LEFT wall target={self.target:.2f} m; '
            f'nominal speed={self.nominal_speed:.3f} m/s'
        )

        self.get_logger().info(f'CSV: {self.csv_path}')

    @staticmethod
    def validate_settings(settings):

        gains = (
            'kp', 'ki', 'kd',
            'kp_linear', 'ki_linear', 'kd_linear'
        )

        speeds = (
            'minimum_speed',
            'nominal_speed',
            'maximum_speed'
        )

        for name in gains + speeds:
            value = settings[name]

            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                return f'{name} must be a finite number'

            if name in gains and not 0.0 <= value <= 10.0:
                return f'{name} must be between 0 and 10'

            if name in speeds and not 0.0 <= value <= 0.12:
                return f'{name} must be between 0 and 0.12 m/s'

        if not (
            settings['minimum_speed']
            <= settings['nominal_speed']
            <= settings['maximum_speed']
        ):
            return (
                'Require minimum_speed <= nominal_speed '
                '<= maximum_speed'
            )

        return ''

    def update_parameters(self, params):

        pending = {
            name: getattr(self, name)
            for name in self.defaults
        }

        changes = {}

        for param in params:
            if param.name in pending:
                pending[param.name] = param.value
                changes[param.name] = param.value

        reason = self.validate_settings(pending)

        if reason:
            return SetParametersResult(
                successful=False,
                reason=reason
            )

        if changes:
            for name, value in changes.items():
                setattr(self, name, float(value))

                self.change_writer.writerow([
                    time.time(),
                    name,
                    float(value)
                ])

            self.change_log.flush()

            # Avoid derivative spikes after parameter changes.
            self.reset_pid()

            self.get_logger().info(
                f'Parameters updated: {changes}'
            )

        return SetParametersResult(successful=True)

    def send(self, linear=0.0, angular=0.0):

        command = Twist()
        command.linear.x = float(linear)
        command.angular.z = float(angular)

        self.pub.publish(command)
        return command

    def reset_pid(self):

        self.error_sum = 0.0
        self.last_error = None
        self.last_stamp = None
        self.filtered_d = 0.0

        self.speed_error_sum = 0.0
        self.last_speed_error = None
        self.last_speed_stamp = None
        self.filtered_speed_d = 0.0

    def change_state(self, new_state):

        if new_state != self.state:
            self.state = new_state
            self.state_since = time.monotonic()
            self.reset_pid()

            self.get_logger().info(
                f'State: {new_state}'
            )

    def fault(self, message):

        if self.state != 'FAULT':
            self.get_logger().error(message)
            self.change_state('FAULT')

        return self.send()

    def odom_callback(self, msg):

        q = msg.pose.pose.orientation

        self.yaw = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        )

        self.measured_speed = msg.twist.twist.linear.x
        self.last_odom = time.monotonic()

    def sensors_ready(self, now):

        return (
            self.last_scan is not None
            and self.last_odom is not None
            and self.yaw is not None
            and self.measured_speed is not None
            and math.isfinite(self.measured_speed)
            and now - self.last_scan <= 1.0
            and now - self.last_odom <= 1.0
        )

    def watchdog(self):

        if not self.sensors_ready(time.monotonic()):
            self.send()
            self.reset_pid()

    @staticmethod
    def distance(scan, angle_deg, width=3.0, nearest=False):

        readings = []

        for i, value in enumerate(scan.ranges):

            if not math.isfinite(value):
                continue

            if not scan.range_min <= value <= scan.range_max:
                continue

            angle = math.degrees(
                scan.angle_min + i * scan.angle_increment
            )

            # Correct wrapped angle; NO arbitrary angular offset.
            difference = (
                angle - angle_deg + 180.0
            ) % 360.0 - 180.0

            if abs(difference) <= width:
                readings.append(value)

        if not readings:
            return None

        return min(readings) if nearest else (
            sum(readings) / len(readings)
        )

    def angular_pid(self, error, stamp):

        if (
            self.last_stamp is not None
            and self.last_error is not None
        ):

            dt = stamp - self.last_stamp

            if 0.001 <= dt <= 1.0:

                self.error_sum = clamp(
                    self.error_sum
                    + 0.5 * (error + self.last_error) * dt,
                    -0.20,
                    0.20
                )

                derivative = (
                    error - self.last_error
                ) / dt

                alpha = dt / (0.30 + dt)

                self.filtered_d += alpha * (
                    derivative - self.filtered_d
                )

            else:
                self.error_sum = 0.0
                self.filtered_d = 0.0

        self.last_error = error
        self.last_stamp = stamp

        command = (
            self.kp * error
            + self.ki * self.error_sum
            + self.kd * self.filtered_d
        )

        return clamp(
            command,
            -self.max_turn_speed,
            self.max_turn_speed
        )

    def reference_speed(self, wall_error, front):

        severity = clamp(
            abs(wall_error) / self.wall_slowdown_error,
            0.0,
            1.0
        )

        target_speed = max(
            self.minimum_speed,
            self.nominal_speed * (1.0 - 0.60 * severity)
        )

        if front < self.front_slowdown_start:

            front_fraction = clamp(
                (front - self.front_limit)
                / (self.front_slowdown_start - self.front_limit),
                0.0,
                1.0
            )

            target_speed = 0.4 + 0.6 * front_fraction

        return clamp(
            target_speed,
            0.0,
            self.nominal_speed
        )

    def linear_pid(self, speed_reference, stamp):

        error = speed_reference - self.measured_speed

        if (
            self.last_speed_stamp is not None
            and self.last_speed_error is not None
        ):

            dt = stamp - self.last_speed_stamp

            if 0.001 <= dt <= 1.0:

                previous_integral = self.speed_error_sum

                self.speed_error_sum = clamp(
                    self.speed_error_sum
                    + 0.5 * (
                        error + self.last_speed_error
                    ) * dt,
                    -0.10,
                    0.10
                )

                derivative = (
                    error - self.last_speed_error
                ) / dt

                alpha = dt / (0.25 + dt)

                self.filtered_speed_d += alpha * (
                    derivative - self.filtered_speed_d
                )

                raw = (
                    speed_reference
                    + self.kp_linear * error
                    + self.ki_linear * self.speed_error_sum
                    + self.kd_linear * self.filtered_speed_d
                )

                if (
                    (raw > self.maximum_speed and error > 0)
                    or (raw < 0.0 and error < 0)
                ):
                    self.speed_error_sum = previous_integral

            else:
                self.speed_error_sum = 0.0
                self.filtered_speed_d = 0.0

        self.last_speed_error = error
        self.last_speed_stamp = stamp

        raw = (
            speed_reference
            + self.kp_linear * error
            + self.ki_linear * self.speed_error_sum
            + self.kd_linear * self.filtered_speed_d
        )

        return clamp(
            raw,
            0.0,
            self.maximum_speed
        )

    def record(
        self, stamp, front, left,
        front_left, cmd, speed_ref=0.0
    ):

        self.writer.writerow([
            stamp,
            front,
            left,
            self.target,
            None if left is None else left - self.target,
            cmd.linear.x,
            cmd.angular.z,
            self.state,
            front_left,
            None if self.yaw is None
            else math.degrees(self.yaw),
            speed_ref,
            self.measured_speed,
            None if self.measured_speed is None
            else speed_ref - self.measured_speed,
            self.kp,
            self.ki,
            self.kd,
            self.kp_linear,
            self.ki_linear,
            self.kd_linear,
            self.minimum_speed,
            self.nominal_speed,
            self.maximum_speed,
        ])

        self.csv_file.flush()

    def scan_callback(self, scan):

        now = time.monotonic()
        self.last_scan = now

        stamp = (
            scan.header.stamp.sec
            + scan.header.stamp.nanosec * 1e-9
        )

        front = self.distance(
            scan, 0, width=12, nearest=True
        )

        left = self.distance(
            scan, 90, width=3
        )

        front_left = self.distance(
            scan, 45, width=12, nearest=True
        )

        front_right = self.distance(
            scan, -45, width=12, nearest=True
        )

        if (
            self.state == 'FAULT'
            or front is None
            or front_left is None
            or front_right is None
            or not self.sensors_ready(now)
        ):

            cmd = self.send()
            self.reset_pid()

            self.record(
                stamp, front, left, front_left, cmd
            )

            return

        if front >= self.front_rearm:
            self.corner_armed = True

        if self.state == 'TURN_RIGHT':

            error = angle_difference(
                self.turn_target, self.yaw
            )

            if now - self.state_since > 20.0:

                cmd = self.fault('Right turn timeout')

            elif front_right < self.front_left_limit:

                # Check direction of clockwise turn (front-right).
                cmd = self.fault(
                    'Obstacle too close during right turn'
                )

            elif abs(math.degrees(error)) <= 2.0:

                self.turn_target = None
                self.change_state('FOLLOW')
                cmd = self.send()

            else:

                cmd = self.send(
                    angular=clamp(
                        1.5 * error, -0.40, 0.40
                    )
                )

            self.record(
                stamp, front, left, front_left, cmd
            )
            return

        if front <= self.front_limit:

            if (
                self.corner_armed
                and front_right >= self.front_left_limit
            ):

                self.corner_armed = False
                self.turn_target = self.yaw - math.pi / 2.0

                self.change_state('TURN_RIGHT')

            cmd = self.send()

            self.record(
                stamp, front, left, front_left, cmd
            )
            return

        if front_left < self.front_left_limit:

            cmd = self.send()
            self.reset_pid()

            self.record(
                stamp, front, left, front_left, cmd
            )
            return

        if self.state == 'FOLLOW':

            if left is None or left > self.max_left:
                self.change_state('RECOVER_LEFT')

            elif left < self.min_left:
                self.change_state('AVOID_LEFT')

        elif self.state == 'RECOVER_LEFT':

            if left is not None and left < self.min_left:
                self.change_state('AVOID_LEFT')

            elif left is not None and left <= 0.40:
                self.change_state('FOLLOW')

        elif self.state == 'AVOID_LEFT':

            if left is None:

                cmd = self.fault(
                    'Lost left LiDAR reading while too close'
                )

                self.record(
                    stamp, front, left, front_left, cmd
                )
                return

            if left >= 0.28:

                self.change_state(
                    'RECOVER_LEFT'
                    if left > self.max_left
                    else 'FOLLOW'
                )

        if self.state in ('RECOVER_LEFT', 'AVOID_LEFT'):

            if now - self.state_since > 8.0:

                cmd = self.fault(
                    'Left-wall recovery timeout'
                )

            elif (
                (
                    self.state == 'RECOVER_LEFT'
                    and front_left < self.front_left_limit
                )
                or (
                    self.state == 'AVOID_LEFT'
                    and front_right < self.front_left_limit
                )
            ):

                cmd = self.fault(
                    'Insufficient clearance for wall recovery turn'
                )

            else:

                turn = (
                    self.recovery_turn_speed
                    if self.state == 'RECOVER_LEFT'
                    else -self.recovery_turn_speed
                )

                cmd = self.send(angular=turn)

        elif self.state == 'FOLLOW' and left is not None:

            error = left - self.target

            angular = self.angular_pid(
                error, stamp
            )

            speed_ref = self.reference_speed(
                error, front
            )

            linear = self.linear_pid(
                speed_ref, stamp
            )

            cmd = self.send(
                linear=linear,
                angular=angular
            )

            self.record(
                stamp,
                front,
                left,
                front_left,
                cmd,
                speed_ref
            )
            return

        else:
            cmd = self.send()

        self.record(
            stamp, front, left, front_left, cmd
        )


def main(args=None):

    rclpy.init(args=args)
    node = None

    try:
        node = ScanReaderLeft()
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:

        if node is not None:
            node.send()
            time.sleep(0.1)

            node.csv_file.close()
            node.change_log.close()
            node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()

