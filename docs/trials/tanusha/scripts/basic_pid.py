#!/usr/bin/env python3
"""ENG612 physical TurtleBot3: improved BASIC SINGLE PID, LEFT wall, RIGHT corners.
Only angular wall-distance control uses PID. Forward speed is rule scheduled.
Experimental controller: supervised bounded real-robot trials only.
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
LEFT_MISSING_TIMEOUT = 0.4
SENSOR_TIMEOUT = 0.6
TURN_TIMEOUT = 12.0
TURN_RATE = 0.30
TURN_TOL_DEG = 3.0
POST_TURN_WAIT = 0.5


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def angular_diff(target, current):
    return math.atan2(math.sin(target-current), math.cos(target-current))


class BasicPID(Node):
    def __init__(self, duration, max_turns):
        super().__init__('eng612_basic_left_improved')
        self.duration, self.max_turns = duration, max_turns
        self.scan, self.odom = None, None
        self.scan_at, self.odom_at = None, None
        self.yaw = None
        self.state = 'DISARMED'
        self.state_at = time.monotonic()
        self.started = None
        self.turn_target = None
        self.turn_count = 0
        self.lost_at = None
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
        self.path = folder / ('basic_left_improved_' + datetime.now().strftime('%Y%m%d_%H%M%S') + '.csv')
        self.file = self.path.open('w', newline='', buffering=1)
        self.writer = csv.writer(self.file)
        self.writer.writerow(['elapsed_s', 'state', 'front_m', 'left_m', 'front_left_m',
            'front_right_m', 'target_m', 'error_m', 'linear_cmd_m_s',
            'angular_cmd_rad_s', 'yaw_deg', 'turn_count', 'kp', 'ki', 'kd'])
        self.get_logger().info('Single angular PID CSV: ' + str(self.path))

    def scan_cb(self, msg):
        self.scan = msg
        self.scan_at = time.monotonic()

    def odom_cb(self, msg):
        self.odom = msg
        self.odom_at = time.monotonic()
        q = msg.pose.pose.orientation
        self.yaw = math.atan2(2*(q.w*q.z+q.x*q.y), 1-2*(q.y*q.y+q.z*q.z))

    def sector(self, angle, width=5, nearest=False):
        if self.scan is None:
            return None
        values=[]
        for i, v in enumerate(self.scan.ranges):
            if not math.isfinite(v) or not self.scan.range_min <= v <= self.scan.range_max:
                continue
            deg=math.degrees(self.scan.angle_min+i*self.scan.angle_increment)
            delta=(deg-angle+180)%360-180
            if abs(delta)<=width:
                values.append(v)
        if not values:
            return None
        return min(values) if nearest else median(values)

    def measures(self):
        return (self.sector(0, 10, True), self.sector(90, 6),
                self.sector(45, 10, True), self.sector(-45, 10, True))

    def send(self, v=0.0, w=0.0):
        t=Twist()
        t.linear.x=float(v)
        t.angular.z=float(w)
        self.pub.publish(t)
        self.last_cmd=t

    def reset_pid(self):
        self.int_e=0.0; self.last_e=None; self.last_t=None; self.d_filter=0.0

    def transition(self, name):
        if name!=self.state:
            self.state=name
            self.state_at=time.monotonic()
            self.reset_pid()
            self.get_logger().info('State: '+name)

    def stop(self, reason):
        if not self.finished:
            self.finished=True
            self.transition('STOPPED')
            self.get_logger().warn('STOP: '+reason)
        self.send()

    def pid(self, error, now):
        derivative=0.0
        if self.last_t is not None and self.last_e is not None:
            dt=now-self.last_t
            if 0.001<dt<=0.5:
                self.int_e=clamp(self.int_e+0.5*(error+self.last_e)*dt, -0.20, 0.20)
                raw=(error-self.last_e)/dt
                alpha=dt/(0.30+dt)
                self.d_filter+=alpha*(raw-self.d_filter)
                derivative=self.d_filter
            else:
                self.reset_pid()
        self.last_t=now
        self.last_e=error
        return clamp(KP*error+KI*self.int_e+KD*derivative, -0.35, 0.35)

    def log(self, front,left,fl,fr):
        if self.started is None:
            return
        error = (left-TARGET) if left is not None else None
        self.writer.writerow([round(time.monotonic()-self.started,3), self.state,
            front,left,fl,fr,TARGET,error,self.last_cmd.linear.x,
            self.last_cmd.angular.z,
            None if self.yaw is None else math.degrees(self.yaw),
            self.turn_count,KP,KI,KD])

    def control(self):
        now=time.monotonic()
        if self.finished:
            self.send()
            return
        if self.started is None:
            self.send()
            return
        if now-self.started>=self.duration:
            self.stop('Duration limit reached')
            return
        if (self.scan_at is None or self.odom_at is None or self.yaw is None
            or now-self.scan_at>SENSOR_TIMEOUT or now-self.odom_at>SENSOR_TIMEOUT):
            self.stop('LiDAR/odometry stale')
            return
        front,left,fl,fr=self.measures()
        if front is None or fl is None or fr is None:
            self.stop('Missing front/corner LiDAR sector')
            return
        if front <= FRONT_HARD or fl <= SIDE_HARD or fr <= SIDE_HARD:
            self.stop('Obstacle too close')
            return
        if self.state=='TURN_RIGHT':
            if now-self.state_at>TURN_TIMEOUT:
                self.stop('Turn timeout')
            else:
                err=angular_diff(self.turn_target,self.yaw)
                if abs(math.degrees(err))<=TURN_TOL_DEG:
                    self.turn_count+=1
                    self.transition('CHECK_NEW_WALL')
                    self.send()
                else:
                    self.send(0.0,clamp(1.4*err,-TURN_RATE,TURN_RATE))
        elif self.state=='CHECK_NEW_WALL':
            self.send()
            if now-self.state_at>=POST_TURN_WAIT:
                if self.turn_count>=self.max_turns:
                    self.stop(f'Completed {self.turn_count} right turn(s)')
                elif left is None or not SIDE_HARD+0.05<=left<=LEFT_VALID_MAX or front<0.55:
                    self.stop('New left wall/front clearance not safe after turn')
                else:
                    self.transition('FOLLOW')
                    self.lost_at=None
        elif self.state=='FOLLOW':
            if left is None or left>LEFT_VALID_MAX:
                self.send()
                if self.lost_at is None:
                    self.lost_at=now
                elif now-self.lost_at>=LEFT_MISSING_TIMEOUT:
                    self.stop('Left wall lost/out of range: operator reposition required')
            elif left < SIDE_HARD+0.05:
                self.stop('Left wall too close')
            else:
                self.lost_at=None
                if front<=FRONT_TURN:
                    self.send()
                    self.turn_target=self.yaw-math.pi/2
                    self.transition('TURN_RIGHT')
                else:
                    # Rule-based speed (not a second PID).
                    speed=CRUISE if front>=SLOW_START else SLOW+(CRUISE-SLOW)*clamp(
                        (front-FRONT_TURN)/(SLOW_START-FRONT_TURN),0.0,1.0)
                    self.send(speed,self.pid(left-TARGET,now))
        self.log(front,left,fl,fr)
        if now-self.last_status>=1.0 and self.state=='FOLLOW':
            self.last_status=now
            self.get_logger().info(f'FOLLOW left={left} front={front:.3f} v={self.last_cmd.linear.x:.3f} turns={self.turn_count}')

    def close(self):
        self.send();time.sleep(0.12);self.send()
        self.file.close()
        self.get_logger().info('CSV saved: '+str(self.path))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--duration',type=float,default=40.0)
    parser.add_argument('--max-turns',type=int,default=1)
    args=parser.parse_args()
    if not 5<=args.duration<=90 or not 1<=args.max_turns<=2:
        parser.error('Use duration 5..90s and max-turns 1..2')
    rclpy.init()
    node=BasicPID(args.duration,args.max_turns)
    try:
        deadline=time.monotonic()+10
        while time.monotonic()<deadline:
            rclpy.spin_once(node,timeout_sec=0.1)
            if node.scan_at is not None and node.odom_at is not None:
                break
        if node.scan_at is None or node.odom_at is None:
            print('ABORT: No fresh scan or odometry')
            return
        front,left,fl,fr=node.measures()
        print(f'Preflight LEFT={left} FRONT={front} FRONT_LEFT={fl} FRONT_RIGHT={fr}')
        if (None in (front,left,fl,fr) or not 0.28<=left<=0.40
            or front<=0.80 or fl<=0.30 or fr<=0.30):
            print('ABORT: Reposition robot, verify wall and clearances')
            return
        if node.count_publishers('/cmd_vel')!=1 or node.count_subscribers('/cmd_vel')!=1:
            print('ABORT: unexpected /cmd_vel publishers/subscribers')
            return
        print('Physical robot: keep cutoff accessible; stop immediately for unexpected motion.')
        if input(f'Type YES for supervised {args.duration:g}s, max {args.max_turns} turn(s): ').strip()!='YES':
            print('Cancelled')
            return
        node.started=time.monotonic()
        node.transition('FOLLOW')
        while rclpy.ok() and not node.finished:
            rclpy.spin_once(node,timeout_sec=0.1)
    except KeyboardInterrupt:
        node.stop('Operator interruption')
    finally:
        node.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__=='__main__':
    main()

