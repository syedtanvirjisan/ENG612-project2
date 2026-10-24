import math
import os
import time
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist


class DataLogger(Node):
    """Logs robot pose and true wall distance to a CSV file."""

    def __init__(self):
        super().__init__('data_logger')
        self.declare_parameter('label', 'run')
        self.declare_parameter('start_x', 0.5)    # robot start in world frame (m)
        self.declare_parameter('start_y', 0.3)
        self.declare_parameter('box_size', 2.0)   # inside size of the square (m)
        self.declare_parameter('out_dir', os.path.expanduser(
            '~/ENG612-Project-2/docs/logs'))

        self.sx = self.get_parameter('start_x').value
        self.sy = self.get_parameter('start_y').value
        self.box = self.get_parameter('box_size').value
        out_dir = self.get_parameter('out_dir').value
        os.makedirs(out_dir, exist_ok=True)
        label = self.get_parameter('label').value
        name = time.strftime(f'{label}_%Y%m%d_%H%M%S.csv')
        self.path = os.path.join(out_dir, name)
        self.file = open(self.path, 'w')
        self.file.write('t,x,y,yaw_deg,v_cmd,w_cmd,true_wall_dist\n')

        self.v_cmd = 0.0
        self.w_cmd = 0.0
        self.t0 = None
        self.create_subscription(Odometry, 'odom', self.odom_cb, 10)
        self.create_subscription(Twist, 'cmd_vel', self.cmd_cb, 10)
        self.get_logger().info(f'Logging to {self.path}')

    def cmd_cb(self, msg):
        self.v_cmd = msg.linear.x
        self.w_cmd = msg.angular.z

    def odom_cb(self, msg):
        t = self.get_clock().now().nanoseconds * 1e-9
        if self.t0 is None:
            self.t0 = t
        # odom starts at (0,0); add the known start pose to get world coords
        x = self.sx + msg.pose.pose.position.x
        y = self.sy + msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                         1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        d_true = min(x, self.box - x, y, self.box - y)
        self.file.write(f'{t - self.t0:.3f},{x:.4f},{y:.4f},'
                        f'{math.degrees(yaw):.2f},{self.v_cmd:.3f},'
                        f'{self.w_cmd:.3f},{d_true:.4f}\n')
        self.file.flush()


def main():
    rclpy.init()
    node = DataLogger()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.file.close()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
