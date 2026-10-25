import math
import time
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


class PID:
    """Simple PID controller with output limits and integral limiting."""

    def __init__(self, kp, ki, kd, out_min, out_max, i_limit):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.out_min, self.out_max = out_min, out_max
        self.i_limit = i_limit
        self.integral = 0.0
        self.prev_error = None

    def step(self, error, dt):
        self.integral = clamp(
            self.integral + error * dt,
            -self.i_limit,
            self.i_limit
        )

        if self.prev_error is None or dt <= 0.0:
            derivative = 0.0
        else:
            derivative = (error - self.prev_error) / dt

        self.prev_error = error

        out = (
            self.kp * error
            + self.ki * self.integral
            + self.kd * derivative
        )

        return clamp(out, self.out_min, self.out_max)

    # NEW: Clear previous PID history after a corner turn
    def reset(self):
        self.integral = 0.0
        self.prev_error = None


class WallFollower(Node):

    def __init__(self):
        super().__init__('wall_follower')

        self.declare_parameter('side', 'right')
        self.declare_parameter('mode', 'extended')
        self.declare_parameter('target', 0.3)

        self.declare_parameter('v_max', 0.12)
        self.declare_parameter('v_min', 0.04)
        self.declare_parameter('lookahead', 0.15)

        self.declare_parameter('kp_w', 4.0)
        self.declare_parameter('ki_w', 0.0)
        self.declare_parameter('kd_w', 1.5)

        self.declare_parameter('kp_v', 0.5)

        self.declare_parameter('front_stop', 0.30)
        self.declare_parameter('front_clear', 1.2)

        self.turning = False

        self.pub = self.create_publisher(
            Twist, 'cmd_vel', 10
        )

        self.sub = self.create_subscription(
            LaserScan,
            'scan',
            self.scan_cb,
            qos_profile_sensor_data
        )

        self.pid_w = None
        self.last_t = None

        self.log = open(
            '/tmp/wall_log.csv',
            'w',
            buffering=1
        )

        self.log.write('t,dist,err,v,w,front\n')

        # ADDED ONLY FOR TILIX: one-second window of wall-distance errors.
        self.stats_count = 0
        self.stats_error_sum = 0.0
        self.stats_abs_error_sum = 0.0
        self.create_timer(1.0, self.print_stats)

        self.get_logger().info('wall_follower started')

    def p(self, name):
        return self.get_parameter(name).value

    # ADDED ONLY FOR TILIX: report averages once per second.
    def print_stats(self):
        target = float(self.p('target'))
        if self.stats_count > 0:
            avg_err_mm = 1000.0 * self.stats_error_sum / self.stats_count
            avg_abs_err_mm = 1000.0 * self.stats_abs_error_sum / self.stats_count
            self.get_logger().info(
                f'[1s] Target: {target:.3f} m | '
                f'Avg signed error: {avg_err_mm:+.2f} mm | '
                f'Avg absolute error: {avg_abs_err_mm:.2f} mm | '
                f'Samples: {self.stats_count}'
            )
        else:
            self.get_logger().info(
                f'[1s] Target: {target:.3f} m | '
                'Avg error: N/A (turning or no wall-following samples)'
            )
        self.stats_count = 0
        self.stats_error_sum = 0.0
        self.stats_abs_error_sum = 0.0

    def beam(self, msg, deg, half=3):
        """Median LiDAR range around the specified angle."""

        n = len(msg.ranges)

        c = int(round(
            (math.radians(deg) - msg.angle_min)
            / msg.angle_increment
        )) % n

        vals = []

        for k in range(-half, half + 1):
            r = msg.ranges[(c + k) % n]

            if not math.isfinite(r) or r < 0.05:
                r = msg.range_max

            vals.append(r)

        vals.sort()

        return vals[len(vals) // 2]

    def scan_cb(self, msg):

        if self.pid_w is None:
            self.pid_w = PID(
                self.p('kp_w'),
                self.p('ki_w'),
                self.p('kd_w'),
                -1.5,
                1.5,
                0.5
            )

        t = self.get_clock().now().nanoseconds * 1e-9

        stamp = (
            msg.header.stamp.sec
            + msg.header.stamp.nanosec * 1e-9
        )

        if self.last_t is not None and stamp <= self.last_t:
            return

        dt = (
            0.0
            if self.last_t is None
            else min(stamp - self.last_t, 0.5)
        )

        self.last_t = stamp

        sign = 1.0 if self.p('side') == 'left' else -1.0

        side_deg = 90 if sign > 0 else 270
        ahead_deg = 45 if sign > 0 else 315

        b = self.beam(msg, side_deg)
        a = self.beam(msg, ahead_deg)

        front = min(
            self.beam(msg, 0, 8),
            self.beam(msg, 15, 4),
            self.beam(msg, 345, 4)
        )

        th = math.radians(45)

        alpha = math.atan2(
            a * math.cos(th) - b,
            a * math.sin(th)
        )

        dist = (
            b * math.cos(alpha)
            + self.p('lookahead') * math.sin(alpha)
        )

        err = dist - self.p('target')

        cmd = Twist()

        if front < self.p('front_stop'):
            self.turning = True

        elif self.turning and front > self.p('front_clear'):

            self.turning = False

            # NEW: Reset derivative and integral history
            # before following the next wall.
            self.pid_w.reset()

            self.get_logger().info(
                'Corner completed: PID state reset'
            )

        if self.turning:

            # Rotate clockwise for left-wall following.
            cmd.angular.z = -sign * 0.8

        elif min(a, b) > 1.0:

            # Search for the wall if it is lost.
            cmd.linear.x = self.p('v_min')
            cmd.angular.z = sign * 0.5

        else:

            cmd.angular.z = (
                sign * self.pid_w.step(err, dt)
            )

            if self.p('mode') == 'baseline':
                cmd.linear.x = self.p('v_max')

            else:

                v_front = clamp(
                    self.p('kp_v')
                    * (front - self.p('front_stop')),
                    0.0,
                    self.p('v_max')
                )

                v_err = (
                    self.p('v_max')
                    * (
                        1.0
                        - min(abs(err) / 0.15, 0.7)
                    )
                )

                cmd.linear.x = clamp(
                    min(v_front, v_err),
                    self.p('v_min'),
                    self.p('v_max')
                )

            # ADDED ONLY FOR TILIX: collect current wall-following error.
            # Skip scans with non-finite or sentinel beam values.
            if (math.isfinite(err)
                    and math.isfinite(a) and math.isfinite(b)
                    and a < msg.range_max and b < msg.range_max):
                self.stats_count += 1
                self.stats_error_sum += err
                self.stats_abs_error_sum += abs(err)

        self.pub.publish(cmd)

        self.log.write(
            f'{t:.3f},{dist:.4f},{err:.4f},'
            f'{cmd.linear.x:.3f},'
            f'{cmd.angular.z:.3f},'
            f'{front:.3f}\n'
        )


def main():

    rclpy.init(
        signal_handler_options=(
            rclpy.signals.SignalHandlerOptions.NO
        )
    )

    node = WallFollower()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:

        try:
            for _ in range(5):
                node.pub.publish(Twist())
                time.sleep(0.05)
        except Exception:
            pass

        node.log.close()
        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
