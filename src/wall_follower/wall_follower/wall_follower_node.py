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
    """PID with clamped integral, filtered derivative and derivative warm-up.

    The warm-up suppresses the derivative term for the first few steps after a
    reset. Without it, the first finite derivative after a corner is computed
    across a large error step and produces a kick on the wheels.
    """

    def __init__(self, kp, ki, kd, out_min, out_max, i_limit,
                 d_alpha=0.3, d_warmup=2):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.out_min, self.out_max = out_min, out_max
        self.i_limit = i_limit
        self.d_alpha = d_alpha
        self.d_warmup = d_warmup
        self.integral = 0.0
        self.prev_error = None
        self.d_filt = 0.0
        self.warmup = d_warmup

    def reset(self):
        self.integral = 0.0
        self.prev_error = None
        self.d_filt = 0.0
        self.warmup = self.d_warmup

    def step(self, error, dt):
        self.integral = clamp(self.integral + error * dt,
                              -self.i_limit, self.i_limit)

        if self.prev_error is None or dt <= 0.0:
            raw = 0.0
        else:
            raw = (error - self.prev_error) / dt
        self.prev_error = error

        if self.warmup > 0:
            self.warmup -= 1
            self.d_filt = 0.0
        else:
            self.d_filt += self.d_alpha * (raw - self.d_filt)

        out = self.kp * error + self.ki * self.integral + self.kd * self.d_filt
        return clamp(out, self.out_min, self.out_max)


# Kept so existing imports/tests that reference LinearPID still work.
LinearPID = PID


class WallFollower(Node):

    PUBLISH_PERIOD = 0.05  # 20 Hz command republish

    def __init__(self):
        super().__init__('wall_follower')

        # ---- original parameters (names and defaults unchanged) ----
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
        self.declare_parameter('ki_v', 0.0)
        self.declare_parameter('kd_v', 0.15)
        self.declare_parameter('c_err', 2.0)
        self.declare_parameter('front_stop', 0.30)
        self.declare_parameter('front_clear', 1.2)

        # ---- new anti-shake parameters ----
        self.declare_parameter('beam_half', 4)          # beams each side of centre
        self.declare_parameter('wall_lost', 1.20)       # enter search above this
        self.declare_parameter('wall_found', 0.90)      # leave search below this
        self.declare_parameter('lost_frames', 3)        # debounce into search
        self.declare_parameter('found_frames', 2)       # debounce out of search
        self.declare_parameter('alpha_max_range', 1.20)  # 45 deg beam trusted below
        self.declare_parameter('alpha_filt', 0.5)       # alpha low-pass (0-1)
        self.declare_parameter('align_tol_deg', 25.0)   # turn exit alignment
        self.declare_parameter('front_resume', 0.60)    # turn exit front clearance
        self.declare_parameter('turn_w', 0.8)           # max turn rate
        self.declare_parameter('turn_w_min', 0.35)      # min turn rate after taper
        self.declare_parameter('recover_time', 0.8)     # gentle window after turn
        self.declare_parameter('w_recover', 0.5)        # angular cap in that window
        self.declare_parameter('w_slew', 1.5)           # rad/s per second
        self.declare_parameter('v_slew', 0.15)          # m/s per second
        self.declare_parameter('d_alpha_w', 0.3)        # angular derivative filter
        self.declare_parameter('jump_limit', 0.25)      # distance outlier gate (m)

        if self.get_parameter('use_sim_time').value:
            self.get_logger().warn(
                'use_sim_time is TRUE. On the real robot launch with '
                'use_sim_time:=false, otherwise the node clock never advances.'
            )

        self.pub = self.create_publisher(Twist, 'cmd_vel', 10)
        self.sub = self.create_subscription(
            LaserScan, 'scan', self.scan_cb, qos_profile_sensor_data
        )

        self.pid_w = None
        self.pid_v = None
        self.last_t = None

        # State machine: 'follow' | 'turn' | 'search'
        self.state = 'follow'
        self.lost_count = 0
        self.found_count = 0
        self.recover_until = 0.0

        # Filtered geometry
        self.alpha_f = 0.0
        self.prev_dist = None
        self.jump_count = 0

        # Command shaping
        self.target_v = 0.0
        self.target_w = 0.0
        self.out_v = 0.0
        self.out_w = 0.0
        self.last_pub_t = None

        self.log = open('/tmp/wall_log.csv', 'w', buffering=1)
        self.log.write('t,dist,err,v,w,front,state\n')

        # Live 1-second statistics (display only; no control changes).
        self.stats_errors = []
        self.stats_window_start = None

        self.timer = self.create_timer(self.PUBLISH_PERIOD, self.publish_cb)
        self.get_logger().info(
            'wall_follower started (dual PID, debounced states, slew-limited)'
        )

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def p(self, name):
        return self.get_parameter(name).value

    def beam(self, msg, deg, half=None):
        """Median of the VALID ranges around an angle (deg, 0=front, CCW+).

        Returns None when no beam in the window is usable. Substituting
        range_max for a dropout, as the original did, injects a large fake
        distance into the median and is a direct cause of sporadic jerks on
        real hardware.
        """
        if half is None:
            half = int(self.p('beam_half'))
        n = len(msg.ranges)
        c = int(round((math.radians(deg) - msg.angle_min) /
                      msg.angle_increment)) % n
        lo = max(msg.range_min, 0.05)
        vals = []
        for k in range(-half, half + 1):
            r = msg.ranges[(c + k) % n]
            if math.isfinite(r) and lo <= r <= msg.range_max:
                vals.append(r)
        if not vals:
            return None
        vals.sort()
        return vals[len(vals) // 2]

    def front_range(self, msg):
        """Smallest valid clearance in the frontal cone."""
        cands = [
            self.beam(msg, 0, 8),
            self.beam(msg, 15, 4),
            self.beam(msg, 345, 4),
        ]
        cands = [c for c in cands if c is not None]
        return min(cands) if cands else msg.range_max

    def enter_follow(self, reason):
        """Any entry into closed-loop following starts from a clean PID."""
        self.state = 'follow'
        self.pid_w.reset()
        self.pid_v.reset()
        self.prev_dist = None
        self.jump_count = 0
        now = self.get_clock().now().nanoseconds * 1e-9
        self.recover_until = now + float(self.p('recover_time'))
        self.get_logger().info(f'{reason}: both PID states reset')

    # ------------------------------------------------------------------
    # 1 s statistics (unchanged output format)
    # ------------------------------------------------------------------
    def report_stats(self, stamp):
        if self.stats_window_start is None:
            self.stats_window_start = stamp
            return

        if stamp - self.stats_window_start < 1.0:
            return

        n = len(self.stats_errors)

        if n:
            avg_signed_m = sum(self.stats_errors) / n
            avg_abs_m = sum(abs(e) for e in self.stats_errors) / n

            self.get_logger().info(
                f'[1s] Target: {self.p("target"):.3f} m | '
                f'Avg signed error: {avg_signed_m:+.6f} m '
                f'({avg_signed_m * 1000:+.2f} mm) | '
                f'Avg absolute error: {avg_abs_m:.6f} m '
                f'({avg_abs_m * 1000:.2f} mm) | '
                f'Samples: {n}'
            )
        else:
            self.get_logger().info(
                f'[1s] Target: {self.p("target"):.3f} m | '
                'Avg error: N/A (turning, searching, or no valid wall samples)'
            )

        self.stats_errors.clear()
        self.stats_window_start = stamp

    # ------------------------------------------------------------------
    # control
    # ------------------------------------------------------------------
    def scan_cb(self, msg):
        if self.pid_w is None:
            self.pid_w = PID(
                self.p('kp_w'), self.p('ki_w'), self.p('kd_w'),
                -1.5, 1.5, 0.5,
                d_alpha=self.p('d_alpha_w'), d_warmup=3
            )
            self.pid_v = PID(
                self.p('kp_v'), self.p('ki_v'), self.p('kd_v'),
                0.0, self.p('v_max'), 0.2,
                d_alpha=0.3, d_warmup=3
            )

        t = self.get_clock().now().nanoseconds * 1e-9
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if self.last_t is not None and stamp <= self.last_t:
            return
        dt = 0.0 if self.last_t is None else min(stamp - self.last_t, 0.5)
        self.last_t = stamp

        sign = 1.0 if self.p('side') == 'left' else -1.0
        side_deg = 90 if sign > 0 else 270
        ahead_deg = 45 if sign > 0 else 315

        b = self.beam(msg, side_deg)
        a = self.beam(msg, ahead_deg)
        front = self.front_range(msg)

        # --- wall angle, guarded -------------------------------------
        # The 45 deg beam frequently misses the wall just after a convex
        # corner. Using it then produces a large bogus alpha and a corrupt
        # distance. Fall back to the perpendicular measurement instead.
        alpha_valid = (
            b is not None and a is not None
            and a < self.p('alpha_max_range')
            and b < self.p('wall_lost')
        )
        if alpha_valid:
            th = math.radians(45)
            alpha_raw = math.atan2(a * math.cos(th) - b, a * math.sin(th))
            alpha_raw = clamp(alpha_raw, -math.radians(70), math.radians(70))
        else:
            alpha_raw = 0.0

        k = clamp(float(self.p('alpha_filt')), 0.0, 1.0)
        self.alpha_f += k * (alpha_raw - self.alpha_f)
        alpha = self.alpha_f

        # --- distance and error --------------------------------------
        wall_valid = b is not None and b < self.p('wall_lost')
        if wall_valid:
            dist = b * math.cos(alpha) + self.p('lookahead') * math.sin(alpha)

            # Reject single-scan outliers; accept after two in a row so a
            # genuine step (doorway, new wall) is not held off indefinitely.
            if (self.prev_dist is not None
                    and abs(dist - self.prev_dist) > self.p('jump_limit')
                    and self.jump_count < 2):
                self.jump_count += 1
                dist = self.prev_dist
            else:
                self.jump_count = 0
            self.prev_dist = dist
        else:
            dist = float('nan')

        err = (dist - self.p('target')) if wall_valid else 0.0

        # --- state machine with hysteresis + debounce ----------------
        if front < self.p('front_stop'):
            if self.state != 'turn':
                self.state = 'turn'
                self.pid_w.reset()
                self.pid_v.reset()
                self.prev_dist = None
            self.lost_count = 0
            self.found_count = 0

        elif self.state == 'turn':
            aligned = (
                wall_valid
                and abs(alpha) < math.radians(self.p('align_tol_deg'))
                and front > self.p('front_resume')
            )
            if aligned or front > self.p('front_clear'):
                self.enter_follow('Corner completed')

        else:
            # follow <-> search, with separate thresholds and frame counts
            if (not wall_valid) or b > self.p('wall_lost'):
                self.lost_count += 1
                self.found_count = 0
            elif b < self.p('wall_found'):
                self.found_count += 1
                self.lost_count = 0
            else:
                self.lost_count = 0
                self.found_count = 0

            if self.state == 'follow' and self.lost_count >= self.p('lost_frames'):
                self.state = 'search'
                self.pid_w.reset()
                self.pid_v.reset()
                self.prev_dist = None
            elif self.state == 'search' and self.found_count >= self.p('found_frames'):
                self.enter_follow('Wall reacquired')

        # --- command generation --------------------------------------
        v = 0.0
        w = 0.0

        if self.state == 'turn':
            # Taper the spin as the front clears instead of a fixed rate.
            span = max(self.p('front_clear') - self.p('front_stop'), 1e-3)
            frac = clamp((self.p('front_clear') - front) / span, 0.0, 1.0)
            w = -sign * clamp(self.p('turn_w') * frac,
                              self.p('turn_w_min'), self.p('turn_w'))
            v = 0.0

        elif self.state == 'search':
            v = self.p('v_min')
            w = sign * 0.5

        else:  # follow
            w = sign * self.pid_w.step(err, dt)

            if self.p('mode') == 'baseline':
                v = self.p('v_max')
            elif self.p('mode') == 'extended_pid':
                e_v = (front - self.p('front_stop')) - self.p('c_err') * abs(err)
                v = clamp(self.pid_v.step(e_v, dt),
                          self.p('v_min'), self.p('v_max'))
            else:
                v_front = clamp(
                    self.p('kp_v') * (front - self.p('front_stop')),
                    0.0, self.p('v_max')
                )
                v_err = self.p('v_max') * (1.0 - min(abs(err) / 0.15, 0.7))
                v = clamp(min(v_front, v_err), self.p('v_min'), self.p('v_max'))

            # Post-turn recovery: re-align gently rather than snapping back.
            if t < self.recover_until:
                w = clamp(w, -self.p('w_recover'), self.p('w_recover'))
                v = self.p('v_min')

        self.target_v = v
        self.target_w = w

        # --- statistics (unchanged gating intent) ---------------------
        if self.state == 'follow' and wall_valid and math.isfinite(err):
            self.stats_errors.append(err)
        self.report_stats(stamp)

        self.log.write(
            f'{t:.3f},{dist if wall_valid else float("nan"):.4f},{err:.4f},'
            f'{self.out_v:.3f},{self.out_w:.3f},{front:.3f},{self.state}\n'
        )

    # ------------------------------------------------------------------
    # 20 Hz slew-limited publishing
    # ------------------------------------------------------------------
    def publish_cb(self):
        now = self.get_clock().now().nanoseconds * 1e-9
        if self.last_pub_t is None:
            dt = self.PUBLISH_PERIOD
        else:
            dt = clamp(now - self.last_pub_t, 1e-3, 0.2)
        self.last_pub_t = now

        max_dw = self.p('w_slew') * dt
        max_dv = self.p('v_slew') * dt

        self.out_w = clamp(self.target_w,
                           self.out_w - max_dw, self.out_w + max_dw)
        self.out_v = clamp(self.target_v,
                           self.out_v - max_dv, self.out_v + max_dv)

        cmd = Twist()
        cmd.linear.x = self.out_v
        cmd.angular.z = self.out_w
        self.pub.publish(cmd)


def main():
    rclpy.init(signal_handler_options=rclpy.signals.SignalHandlerOptions.NO)
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
