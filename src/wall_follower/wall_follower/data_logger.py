import csv
import math
import os
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist


class DataLogger(Node):
    """Log LiDAR-estimated left/right wall clearance; no map-frame assumptions."""

    def __init__(self):
        super().__init__('data_logger')
        self.declare_parameter('label', 'real_extended')
        self.declare_parameter('out_dir', os.path.expanduser('~/ENG612_New_Logs'))
        self.declare_parameter('target', 0.30)
        self.declare_parameter('side', 'left')
        self.declare_parameter('lookahead', 0.15)
        self.declare_parameter('max_wall_range', 1.0)
        self.declare_parameter('corner_cooldown', 1.0)

        self.target = float(self.get_parameter('target').value)
        self.side = str(self.get_parameter('side').value)
        self.lookahead = float(self.get_parameter('lookahead').value)
        self.max_wall_range = float(self.get_parameter('max_wall_range').value)
        self.corner_cooldown = float(self.get_parameter('corner_cooldown').value)
        if self.side not in ('left', 'right'):
            raise ValueError('side must be left or right')

        out_dir = os.path.expanduser(str(self.get_parameter('out_dir').value))
        os.makedirs(out_dir, exist_ok=True)
        label = str(self.get_parameter('label').value)
        safe_label = ''.join(c if c.isalnum() or c in '-_' else '_' for c in label)
        self.path = os.path.join(out_dir, time.strftime(f'{safe_label}_%Y%m%d_%H%M%S.csv'))
        self.file = open(self.path, 'w', newline='', buffering=1)
        self.writer = csv.writer(self.file)
        self.writer.writerow(['t_s', 'odom_x_m', 'odom_y_m', 'yaw_deg',
                              'v_cmd_m_s', 'w_cmd_rad_s', 'side_beam_m',
                              'ahead_beam_m', 'front_m', 'lidar_wall_dist_m',
                              'target_m', 'error_m', 'valid_straight'])

        self.t0 = None
        self.x = float('nan')
        self.y = float('nan')
        self.yaw = float('nan')
        self.v_cmd = 0.0
        self.w_cmd = 0.0
        self.have_cmd = False
        self.last_corner_time = None
        self.n_scans = 0
        self.errors = []
        self.create_subscription(Odometry, 'odom', self.odom_cb, 10)
        self.create_subscription(Twist, 'cmd_vel', self.cmd_cb, 10)
        self.create_subscription(LaserScan, 'scan', self.scan_cb, qos_profile_sensor_data)
        self.get_logger().info(f'LiDAR logging to {self.path}')

    def odom_cb(self, msg):
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        self.yaw = math.degrees(math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z)))

    def cmd_cb(self, msg):
        self.v_cmd = msg.linear.x
        self.w_cmd = msg.angular.z
        self.have_cmd = True
        if abs(self.v_cmd) < 0.005 and abs(self.w_cmd) > 0.5:
            self.last_corner_time = self.get_clock().now().nanoseconds * 1e-9

    @staticmethod
    def beam(msg, deg, half=3):
        """Same median-beam convention as the current wall_follower_node.py."""
        n = len(msg.ranges)
        if n == 0 or msg.angle_increment == 0:
            return float('nan')
        angle = math.radians(deg)
        # Map angles to the scan's native interval (supports [-pi,pi] and [0,2pi]).
        while angle < msg.angle_min:
            angle += 2.0 * math.pi
        while angle > msg.angle_max:
            angle -= 2.0 * math.pi
        center = round((angle - msg.angle_min) / msg.angle_increment)
        values = []
        for k in range(-half, half + 1):
            idx = int(center) + k
            # Use only angles actually covered by the sensor scan.
            if idx < 0 or idx >= n:
                continue
            r = msg.ranges[idx]
            if not math.isfinite(r) or r < max(msg.range_min, 0.05):
                r = msg.range_max
            values.append(r)
        return sorted(values)[len(values) // 2] if values else float('nan')

    def scan_cb(self, msg):
        now = self.get_clock().now().nanoseconds * 1e-9
        if self.t0 is None:
            self.t0 = now
        self.n_scans += 1
        left = self.side == 'left'
        b = self.beam(msg, 90 if left else 270)
        a = self.beam(msg, 45 if left else 315)
        front = min(self.beam(msg, 0, 8),
                    self.beam(msg, 15, 4),
                    self.beam(msg, 345, 4))

        if math.isfinite(a) and math.isfinite(b):
            th = math.radians(45)
            alpha = math.atan2(a * math.cos(th) - b, a * math.sin(th))
            dist = b * math.cos(alpha) + self.lookahead * math.sin(alpha)
        else:
            dist = float('nan')
        err = dist - self.target if math.isfinite(dist) else float('nan')

        in_cooldown = (self.last_corner_time is not None and
                       now - self.last_corner_time < self.corner_cooldown)
        # Do not include obvious corner rotations, wall-lost searches, or invalid beams.
        valid = (self.have_cmd and self.v_cmd > 0.005 and
                 abs(self.w_cmd) < 0.5 and not in_cooldown and
                 math.isfinite(front) and math.isfinite(dist) and
                 0.05 <= a <= self.max_wall_range and
                 0.05 <= b <= self.max_wall_range and
                 0.05 <= dist <= self.max_wall_range)
        if valid:
            self.errors.append(err)

        self.writer.writerow([
            round(now - self.t0, 3), round(self.x, 4), round(self.y, 4),
            round(self.yaw, 2), round(self.v_cmd, 3), round(self.w_cmd, 3),
            round(b, 4), round(a, 4), round(front, 4), round(dist, 4),
            round(self.target, 4), round(err, 4), int(valid)
        ])

    def save_summary(self):
        path = self.path.removesuffix('.csv') + '_summary.csv'
        n = len(self.errors)
        metrics = [
            ('Target Distance', self.target, 'm'),
            ('Total LiDAR Samples', self.n_scans, 'samples'),
            ('Valid Straight-Wall Samples', n, 'samples'),
            ('Excluded Samples', self.n_scans - n, 'samples'),
        ]
        if n:
            mean = sum(self.errors) / n
            mae = sum(abs(e) for e in self.errors) / n
            mse = sum(e * e for e in self.errors) / n
            metrics += [
                ('Mean Error', mean, 'm'),
                ('Mean Absolute Error', mae, 'm'),
                ('Mean Squared Error', mse, 'm^2'),
                ('Root Mean Squared Error', math.sqrt(mse), 'm'),
                ('Maximum Absolute Error', max(abs(e) for e in self.errors), 'm'),
            ]
        else:
            self.get_logger().warn('No eligible samples: check scan angles, side, parameters and cmd_vel.')
        with open(path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Metric', 'Value', 'Unit'])
            writer.writerows(metrics)
        for key, value, unit in metrics:
            self.get_logger().info(f'{key}: {value:.6f} {unit}' if isinstance(value, float)
                                   else f'{key}: {value} {unit}')
        self.get_logger().info(f'Summary saved to {path}')


def main():
    rclpy.init()
    node = DataLogger()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.save_summary()
        node.file.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()

