#!/usr/bin/env python3
"""ENG612 Basic SINGLE angular PID: LEFT wall following with LiDAR wall angle.

Turns RIGHT by odometry, then aligns to the new LEFT wall before following.
No second speed PID: forward speed is scheduled by clearance/error.
Experimental real-robot controller: supervised short tests only.
"""
import argparse
import csv
import math
import time
from datetime import datetime
from pathlib import Path
from statistics import median

import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

TARGET = 0.30
KP, KI, KD = 1.0, 0.01, 0.50
CRUISE = 0.05
SLOW = 0.018
SLOW_START = 0.65
FRONT_TURN = 0.35
FRONT_HARD = 0.24
SIDE_HARD = 0.20
LEFT_VALID_MAX = 0.45
LEFT_MISSING_TIMEOUT = 0.40
SENSOR_TIMEOUT = 0.60
TURN_TIMEOUT = 12.0
TURN_RATE = 0.30
TURN_TOL_DEG = 3.0
POST_TURN_WAIT = 0.50
ALIGN_TIMEOUT = 4.0
ALIGN_RATE = 0.18
ALIGN_TOL_DEG = 6.0
ALIGN_HOLD = 0.35
LOOKAHEAD = 0.12  # metres: convert wall heading into equivalent distance correction
MAX_HEADING_DEG = 25.0


def clamp(value, lo, hi):
    return max(lo, min(hi, value))


def angular_diff(target, current):
    return math.atan2(math.sin(target - current), math.cos(target - current))


class BasicPID(Node):
    def __init__(self, duration, max_turns):
        super().__init__('eng612_basic_left_wall_angle')
        self.duration = duration
        self.max_turns = max_turns
        self.scan = None
        self.odom = None
        self.scan_at = None
        self.odom_at = None
        self.yaw = None
        self.state = 'DISARMED'
        self.state_at = time.monotonic()
        self.started = None
        self.turn_target = None
        self.turn_count = 0
        self.lost_at = None
        self.align_good_at = None
        self.finished = False
        self.last_status = 0.0
        self.last_cmd = Twist()
        self.int_e = 0.0
        self.last_e = None
        self.last_t = None
        self.d_filter = 0.0
        self.pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.create_subscription(LaserScan, '/scan', self.scan_cb, qos_profile_sensor_data)
        self.create_subscription(Odometry, '/odom', self.odom_cb, qos_profile_sensor_data)
        self.create_timer(0.1, self.control)
        folder = Path.home() / 'wall_follower_logs'
        folder.mkdir(parents=True, exist_ok=True)
        self.path = folder / ('basic_left_angle_' + datetime.now().strftime('%Y%m%d_%H%M%S') + '.csv')
        self.file = self.path.open('w', newline='', buffering=1)
        self.writer = csv.writer(self.file)
        self.writer.writerow([
            'elapsed_s', 'state', 'front_m', 'left_m', 'front_left_m',
            'front_right_m', 'target_m', 'distance_error_m', 'wall_angle_deg',
            'effective_error_m', 'linear_cmd_m_s', 'angular_cmd_rad_s',
            'yaw_deg', 'turn_count', 'kp', 'ki', 'kd'
        ])
        self.get_logger().info('Single PID log: ' + str(self.path))

    def scan_cb(self, msg):
        self.scan = msg
        self.scan_at = time.monotonic()

    def odom_cb(self, msg):
        self.odom = msg
        self.odom_at = time.monotonic()
        q = msg.pose.pose.orientation
        self.yaw = math.atan2(2 * (q.w*q.z + q.x*q.y),
                              1 - 2 * (q.y*q.y + q.z*q.z))

    def sector(self, angle, width=5, nearest=False):
        if self.scan is None:
            return None
        values = []
        for i, reading in enumerate(self.scan.ranges):
            if not (math.isfinite(reading) and self.scan.range_min <= reading <= self.scan.range_max):
                continue
            deg = math.degrees(self.scan.angle_min + i * self.scan.angle_increment)
            delta = (deg - angle + 180) % 360 - 180
            if abs(delta) <= width:
                values.append(reading)
        if not values:
            return None
        return min(values) if nearest else median(values)

    def measures(self):
        return (self.sector(0, 10, True), self.sector(90, 6),
                self.sector(45, 10, True), self.sector(-45, 10, True))

    def wall_angle(self):
        """Estimate left-wall heading using LiDAR at 70 and 110 degrees.

        +angle => wall runs away toward robot's front => steer LEFT (+w).
        Avoid using this estimate when sectors likely see different walls.
        """
        forward_r = self.sector(70, 3)
        back_r = self.sector(110, 3)
        if forward_r is None or back_r is None:
            return None
        # Reject sectors far from the expected wall before fitting a line.
        if not (0.18 <= forward_r <= 0.70 and 0.18 <= back_r <= 0.70):
            return None
        a = math.radians(70)
        b = math.radians(110)
        xf, yf = forward_r * math.cos(a), forward_r * math.sin(a)
        xb, yb = back_r * math.cos(b), back_r * math.sin(b)
        theta = math.atan2(yf - yb, xf - xb)
        if abs(math.degrees(theta)) > MAX_HEADING_DEG:
            return None
        return theta

    def send(self, v=0.0, w=0.0):
        msg = Twist()
        msg.linear.x = float(v)
        msg.angular.z = float(w)
        self.pub.publish(msg)
        self.last_cmd = msg

    def reset_pid(self):
        self.int_e = 0.0
        self.last_e = None
        self.last_t = None
        self.d_filter = 0.0

    def transition(self, new_state):
        if new_state != self.state:
            self.state = new_state
            self.state_at = time.monotonic()
            self.align_good_at = None
            self.reset_pid()
            self.get_logger().info('State: ' + new_state)

    def stop(self, reason):
        if not self.finished:
            self.finished = True
            self.transition('STOPPED')
            self.get_logger().warn('STOP: ' + reason)
        self.send()

    def pid(self, error, now):
        derivative = 0.0
        if self.last_t is not None and self.last_e is not None:
            dt = now - self.last_t
            if 0.001 < dt <= 0.5:
                self.int_e = clamp(self.int_e + 0.5 * (error + self.last_e) * dt,
                                   -0.20, 0.20)
                raw = (error - self.last_e) / dt
                alpha = dt / (0.30 + dt)
                self.d_filter += alpha * (raw - self.d_filter)
                derivative = self.d_filter
            else:
                self.reset_pid()
        self.last_t = now
        self.last_e = error
        return clamp(KP * error + KI * self.int_e + KD * derivative, -0.35, 0.35)

    def log(self, front, left, fl, fr, heading, effective_error):
        if self.started is None:
            return
        self.writer.writerow([
            round(time.monotonic() - self.started, 3), self.state,
            front, left, fl, fr, TARGET,
            None if left is None else left - TARGET,
            None if heading is None else math.degrees(heading),
            effective_error, self.last_cmd.linear.x, self.last_cmd.angular.z,
            None if self.yaw is None else math.degrees(self.yaw),
            self.turn_count, KP, KI, KD
        ])

    def control(self):
        now = time.monotonic()
        if self.finished or self.started is None:
            self.send()
            return
        if now - self.started >= self.duration:
            self.stop('Duration limit reached')
            return
        if (self.scan_at is None or self.odom_at is None or self.yaw is None
                or now - self.scan_at > SENSOR_TIMEOUT
                or now - self.odom_at > SENSOR_TIMEOUT):
            self.stop('LiDAR/odometry stale')
            return

        front, left, fl, fr = self.measures()
        heading = self.wall_angle()
        effective_error = None
        if front is None or fl is None or fr is None:
            self.stop('Missing front/corner LiDAR sector')
            return
        if front <= FRONT_HARD or fl <= SIDE_HARD or fr <= SIDE_HARD:
            self.stop('Obstacle too close')
            return
        # Immediate stop if the left wall is dangerously close while not mid-turn.
        if self.state != 'TURN_RIGHT' and left is not None and left < SIDE_HARD + 0.05:
            self.stop('Left wall too close')
            return

        if self.state == 'TURN_RIGHT':
            if now - self.state_at > TURN_TIMEOUT:
                self.stop('Turn timeout')
            else:
                err = angular_diff(self.turn_target, self.yaw)
                if abs(math.degrees(err)) <= TURN_TOL_DEG:
                    self.turn_count += 1
                    self.transition('CHECK_NEW_WALL')
                    self.send()
                else:
                    self.send(0.0, clamp(1.4 * err, -TURN_RATE, TURN_RATE))

        elif self.state == 'CHECK_NEW_WALL':
            self.send()
            if now - self.state_at >= POST_TURN_WAIT:
                if self.turn_count >= self.max_turns:
                    self.stop(f'Completed {self.turn_count} right turn(s)')
                elif (left is None or not 0.25 <= left <= LEFT_VALID_MAX
                      or front < 0.55 or heading is None):
                    self.stop('New left wall/angle/front clearance invalid after turn')
                else:
                    self.transition('ALIGN_LEFT')

        elif self.state == 'ALIGN_LEFT':
            if now - self.state_at >= ALIGN_TIMEOUT:
                self.stop('Post-turn wall alignment timeout')
            elif left is None or not 0.25 <= left <= LEFT_VALID_MAX or front < 0.55 or heading is None:
                self.stop('Unsafe/missing wall during post-turn alignment')
            elif abs(math.degrees(heading)) <= ALIGN_TOL_DEG:
                self.send()
                if self.align_good_at is None:
                    self.align_good_at = now
                elif now - self.align_good_at >= ALIGN_HOLD:
                    self.transition('FOLLOW')
                    self.lost_at = None
            else:
                self.align_good_at = None
                # Rotation in place only: no forward motion during alignment.
                self.send(0.0, clamp(1.2 * heading, -ALIGN_RATE, ALIGN_RATE))

        elif self.state == 'FOLLOW':
            if left is None or left > LEFT_VALID_MAX:
                self.send()
                if self.lost_at is None:
                    self.lost_at = now
                elif now - self.lost_at >= LEFT_MISSING_TIMEOUT:
                    self.stop('Left wall lost/out of range: operator reposition required')
            else:
                self.lost_at = None
                if front <= FRONT_TURN:
                    self.send()
                    self.turn_target = self.yaw - math.pi / 2
                    self.transition('TURN_RIGHT')
                else:
                    # ONE angular PID; angle contributes to its error only.
                    effective_error = left - TARGET
                    if heading is not None:
                        effective_error += LOOKAHEAD * math.sin(heading)
                    speed = (CRUISE if front >= SLOW_START else
                             SLOW + (CRUISE - SLOW) * clamp(
                                 (front - FRONT_TURN) / (SLOW_START - FRONT_TURN), 0.0, 1.0))
                    # Lower forward speed when steering correction is unusually large.
                    if abs(effective_error) > 0.06:
                        speed = min(speed, 0.03)
                    self.send(speed, self.pid(effective_error, now))

        self.log(front, left, fl, fr, heading, effective_error)
        if now - self.last_status >= 1.0 and self.state == 'FOLLOW':
            self.last_status = now
            angle_text = 'NA' if heading is None else f'{math.degrees(heading):+.1f}'
            self.get_logger().info(
                f'FOLLOW left={left} angle={angle_text}deg front={front:.3f} '
                f'v={self.last_cmd.linear.x:.3f} turns={self.turn_count}')

    def close(self):
        self.send()
        time.sleep(0.12)
        self.send()
        self.file.close()
        self.get_logger().info('CSV saved: ' + str(self.path))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--duration', type=float, default=40.0)
    parser.add_argument('--max-turns', type=int, default=1)
    args = parser.parse_args()
    if not 5 <= args.duration <= 90 or not 1 <= args.max_turns <= 2:
        parser.error('Use duration 5..90s and max-turns 1..2')
    rclpy.init()
    node = BasicPID(args.duration, args.max_turns)
    try:
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.1)
            if node.scan_at is not None and node.odom_at is not None:
                break
        if (node.scan_at is None or node.odom_at is None
                or time.monotonic() - node.scan_at > SENSOR_TIMEOUT
                or time.monotonic() - node.odom_at > SENSOR_TIMEOUT):
            print('ABORT: No fresh scan or odometry')
            return
        front, left, fl, fr = node.measures()
        heading = node.wall_angle()
        print(f'Preflight LEFT={left} FRONT={front} FRONT_LEFT={fl} '
              f'FRONT_RIGHT={fr} WALL_ANGLE_DEG='
              f'{None if heading is None else round(math.degrees(heading), 1)}')
        if (None in (front, left, fl, fr, heading)
                or not 0.28 <= left <= 0.40
                or abs(math.degrees(heading)) > 15
                or front <= 0.80 or fl <= 0.30 or fr <= 0.30):
            print('ABORT: Reposition robot; verify LEFT wall/angle and clearances')
            return
        if node.count_publishers('/cmd_vel') != 1 or node.count_subscribers('/cmd_vel') != 1:
            print('ABORT: unexpected /cmd_vel publishers/subscribers')
            return
        print('Physical robot: keep cutoff accessible; stop immediately for unexpected motion.')
        if input(f'Type YES for supervised {args.duration:g}s, max {args.max_turns} turn(s): ').strip() != 'YES':
            print('Cancelled')
            return
        # input() blocks callbacks; collect several NEW LiDAR scans after YES.
        # Stay disarmed, publish zero speed, and never accept stale readings.
        node.scan_at = None
        node.odom_at = None
        refresh_deadline = time.monotonic() + 6.0
        last_checked_scan = None
        consecutive_good = 0
        accepted = False
        reason = 'No fresh LiDAR/odometry'
        while time.monotonic() < refresh_deadline:
            node.send()
            rclpy.spin_once(node, timeout_sec=0.1)
            now = time.monotonic()
            if (node.scan_at is None or node.odom_at is None
                    or now - node.scan_at > SENSOR_TIMEOUT
                    or now - node.odom_at > SENSOR_TIMEOUT):
                consecutive_good = 0
                continue
            # Evaluate only once per actual scan, not repeated timer cycles.
            if node.scan_at == last_checked_scan:
                continue
            last_checked_scan = node.scan_at
            front, left, fl, fr = node.measures()
            heading = node.wall_angle()
            heading_deg = None if heading is None else math.degrees(heading)
            # Diagnostic sectors show exactly why the angle may be unavailable.
            r70, r110 = node.sector(70, 3), node.sector(110, 3)
            print(f'Updated scan LEFT={left} FRONT={front} '
                  f'FL={fl} FR={fr} ANGLE={heading_deg} '
                  f'L70={r70} L110={r110}')
            # Any immediately dangerous clearance reading aborts, not filters.
            if (front is not None and front <= FRONT_HARD
                    or fl is not None and fl <= SIDE_HARD
                    or fr is not None and fr <= SIDE_HARD
                    or left is not None and left < SIDE_HARD + 0.05):
                reason = 'Obstacle or left wall too close'
                break
            good = (None not in (front, left, fl, fr, heading)
                    and 0.28 <= left <= 0.40
                    and abs(heading_deg) <= 15.0
                    and front > 0.80 and fl > 0.30 and fr > 0.30)
            if good:
                consecutive_good += 1
                if consecutive_good >= 3:
                    accepted = True
                    break
            else:
                consecutive_good = 0
                reason = ('Intermittent/invalid wall-angle or preflight clearance; '
                          'check L70/L110 and robot alignment')
        if not accepted:
            print('ABORT: ' + reason)
            return
        if node.count_publishers('/cmd_vel') != 1 or node.count_subscribers('/cmd_vel') != 1:
            print('ABORT: unexpected /cmd_vel publishers/subscribers')
            return
        print('Preflight passed: 3 consecutive valid fresh LiDAR scans')
        node.started = time.monotonic()
        node.transition('FOLLOW')
        while rclpy.ok() and not node.finished:
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        node.stop('Operator interruption')
    finally:
        node.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()

