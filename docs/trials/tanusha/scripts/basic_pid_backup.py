"""ENG612 Basic PID: TurtleBot3 Burger LEFT-wall follower.

Single PID controls angular velocity; forward velocity is constant in FOLLOW.
Requires /scan (LaserScan) and /odom (Odometry). Test in Gazebo first.
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


def clamp(x, low, high):
    return max(low, min(high, x))


def angular_difference(target, actual):
    return math.atan2(math.sin(target - actual), math.cos(target - actual))


class BasicLeftWallFollower(Node):
    def __init__(self):
        super().__init__('basic_left_wall_pid')
        self.target = 0.30
        self.speed = 0.05
        self.front_limit = 0.35
        self.front_rearm = 0.48
        self.front_right_limit = 0.20
        self.front_left_limit = 0.20
        self.min_left = 0.20
        self.max_left = 0.50
        self.recover_exit = 0.43
        self.avoid_exit = 0.26
        self.max_angular = 0.50
        self.recovery_angular = 0.22
        self.turn_angular = 0.40

        for name, initial in (('kp', 1.0), ('ki', 0.01), ('kd', 0.50)):
            self.declare_parameter(name, initial)
            setattr(self, name, float(self.get_parameter(name).value))
        self.add_on_set_parameters_callback(self.update_gains)

        self.pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.create_subscription(LaserScan, '/scan', self.on_scan, qos_profile_sensor_data)
        self.create_subscription(Odometry, '/odom', self.on_odom, qos_profile_sensor_data)

        self.scan = None
        self.yaw = None
        self.scan_received = None
        self.odom_received = None
        self.state = 'FOLLOW'
        self.state_started = time.monotonic()
        self.corner_armed = True
        self.turn_target = None
        self.reset_pid()

        folder = Path.home() / 'wall_follower_logs'
        folder.mkdir(parents=True, exist_ok=True)
        filename = 'basic_left_' + datetime.now().strftime('%Y%m%d_%H%M%S') + '.csv'
        self.csv_file = (folder / filename).open('w', newline='', encoding='utf-8')
        self.csv = csv.writer(self.csv_file)
        self.csv.writerow(['time_s', 'state', 'front_m', 'left_m', 'front_left_m',
                           'front_right_m', 'target_m', 'error_m', 'linear_cmd_m_s',
                           'angular_cmd_rad_s', 'yaw_deg', 'kp', 'ki', 'kd'])
        self.create_timer(0.1, self.control)
        self.get_logger().info(f'BASIC LEFT PID ready; log: {folder / filename}')

    def reset_pid(self):
        self.integral = 0.0
        self.previous_error = None
        self.previous_time = None
        self.filtered_d = 0.0

    def update_gains(self, params):
        updates = {}
        for p in params:
            if p.name in ('kp', 'ki', 'kd'):
                value = p.value
                if (isinstance(value, bool) or not isinstance(value, (int, float))
                        or not math.isfinite(float(value)) or not 0 <= value <= 10):
                    return SetParametersResult(successful=False,
                                               reason=f'{p.name} must be between 0 and 10')
                updates[p.name] = float(value)
        for name, value in updates.items():
            setattr(self, name, value)
        if updates:
            self.reset_pid()
            self.get_logger().info(f'Updated PID gains: {updates}')
        return SetParametersResult(successful=True)

    def on_scan(self, msg):
        self.scan = msg
        self.scan_received = time.monotonic()

    def on_odom(self, msg):
        q = msg.pose.pose.orientation
        self.yaw = math.atan2(2 * (q.w * q.z + q.x * q.y),
                              1 - 2 * (q.y * q.y + q.z * q.z))
        self.odom_received = time.monotonic()

    def distance(self, direction, half_width=5, nearest=False):
        if self.scan is None:
            return None
        readings = []
        for index, value in enumerate(self.scan.ranges):
            if not math.isfinite(value) or not self.scan.range_min <= value <= self.scan.range_max:
                continue
            angle = math.degrees(self.scan.angle_min + index * self.scan.angle_increment)
            delta = (angle - direction + 180) % 360 - 180
            if abs(delta) <= half_width:
                readings.append(value)
        if not readings:
            return None
        return min(readings) if nearest else sum(readings) / len(readings)

    def command(self, linear=0.0, angular=0.0):
        message = Twist()
        message.linear.x = float(linear)
        message.angular.z = float(angular)
        self.pub.publish(message)
        return message

    def transition(self, new_state):
        if self.state != new_state:
            self.state = new_state
            self.state_started = time.monotonic()
            self.reset_pid()
            self.get_logger().info(f'State: {new_state}')

    def fault(self, reason):
        self.get_logger().error(reason)
        self.transition('FAULT')
        return self.command()

    def angular_pid(self, error, now):
        derivative = 0.0
        if self.previous_time is not None and self.previous_error is not None:
            dt = now - self.previous_time
            if 0.001 <= dt <= 0.5:
                self.integral = clamp(self.integral + (error + self.previous_error) * dt / 2,
                                      -0.20, 0.20)
                raw_d = (error - self.previous_error) / dt
                alpha = dt / (0.30 + dt)
                self.filtered_d += alpha * (raw_d - self.filtered_d)
                derivative = self.filtered_d
            else:
                self.reset_pid()
        self.previous_time = now
        self.previous_error = error
        # Left wall: positive error => turn LEFT (positive angular.z).
        return clamp(self.kp * error + self.ki * self.integral + self.kd * derivative,
                     -self.max_angular, self.max_angular)

    def log(self, now, front, left, front_left, front_right, cmd):
        self.csv.writerow([round(now, 3), self.state, front, left, front_left,
                           front_right, self.target,
                           None if left is None else left - self.target,
                           cmd.linear.x, cmd.angular.z,
                           None if self.yaw is None else math.degrees(self.yaw),
                           self.kp, self.ki, self.kd])
        self.csv_file.flush()

    def control(self):
        now = time.monotonic()
        if (self.scan_received is None or self.odom_received is None
                or now - self.scan_received > 0.5 or now - self.odom_received > 0.5):
            self.command()
            self.reset_pid()
            return

        front = self.distance(0, 10, nearest=True)
        left = self.distance(90, 5)
        front_left = self.distance(45, 10, nearest=True)
        front_right = self.distance(-45, 10, nearest=True)
        cmd = Twist()  # Safe default; publish only one command per cycle.

        if self.state == 'FAULT' or any(v is None for v in (front, front_left, front_right)):
            cmd = self.command()
            self.reset_pid()
            self.log(now, front, left, front_left, front_right, cmd)
            return

        if front >= self.front_rearm:
            self.corner_armed = True

        if self.state == 'TURN_RIGHT':
            if now - self.state_started > 12.0:
                cmd = self.fault('Corner turn timed out')
            elif front_right < self.front_right_limit:
                cmd = self.fault('Obstacle too close on right during turn')
            else:
                yaw_error = angular_difference(self.turn_target, self.yaw)
                if abs(yaw_error) <= math.radians(3):
                    self.turn_target = None
                    self.transition('FOLLOW')
                    cmd = self.command()
                else:
                    cmd = self.command(angular=clamp(1.5 * yaw_error,
                                                     -self.turn_angular, self.turn_angular))
            self.log(now, front, left, front_left, front_right, cmd)
            return

        # Front wall has priority over wall recovery.
        if front <= self.front_limit:
            if self.corner_armed and front_right >= self.front_right_limit:
                self.corner_armed = False
                self.turn_target = self.yaw - math.pi / 2
                self.transition('TURN_RIGHT')
            cmd = self.command()
            self.log(now, front, left, front_left, front_right, cmd)
            return

        if front_left < self.front_left_limit:
            cmd = self.command()
            self.reset_pid()
            self.log(now, front, left, front_left, front_right, cmd)
            return

        if self.state == 'FOLLOW':
            if left is None or left > self.max_left:
                self.transition('RECOVER_LEFT')
            elif left < self.min_left:
                self.transition('AVOID_LEFT')
        elif self.state == 'RECOVER_LEFT':
            if left is not None and left < self.min_left:
                self.transition('AVOID_LEFT')
            elif left is not None and left <= self.recover_exit:
                self.transition('FOLLOW')
        elif self.state == 'AVOID_LEFT':
            if left is None:
                cmd = self.fault('Lost left wall while too close')
            elif left >= self.avoid_exit:
                self.transition('FOLLOW' if left <= self.max_left else 'RECOVER_LEFT')

        if self.state in ('RECOVER_LEFT', 'AVOID_LEFT'):
            if now - self.state_started > 8.0:
                cmd = self.fault('Wall recovery timed out')
            elif front_right < self.front_right_limit:
                cmd = self.fault('Right-side clearance too small to recover')
            else:
                # Lost left wall -> rotate LEFT; too close -> rotate RIGHT.
                angular = (self.recovery_angular if self.state == 'RECOVER_LEFT'
                           else -self.recovery_angular)
                cmd = self.command(angular=angular)
        elif self.state == 'FOLLOW' and left is not None:
            error = left - self.target
            cmd = self.command(linear=self.speed, angular=self.angular_pid(error, now))
            self.get_logger().info(
                f'Left={left:.3f} m error={error:+.3f} m',
                throttle_duration_sec=1.0)

        if self.state not in ('FOLLOW', 'RECOVER_LEFT', 'AVOID_LEFT', 'FAULT'):
            cmd = self.command()
        self.log(now, front, left, front_left, front_right, cmd)

    def close(self):
        for _ in range(3):
            self.command()
            time.sleep(0.05)
        self.csv_file.close()


def main(args=None):
    rclpy.init(args=args)
    node = BasicLeftWallFollower()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
