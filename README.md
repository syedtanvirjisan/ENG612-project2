ENG612 Project 2 – TurtleBot3 Burger Wall Following Robot

Student: Syed Tanvir
University: Murdoch University, Perth, Australia
Course: Master of Intelligence Industrial Control and Autonomous System Engineering
Unit: ENG612 – Applied Robotics
Project: LiDAR-Based Autonomous Wall Following Using PID Control

1. Project Overview

This project develops and evaluates an autonomous wall-following system using a physical TurtleBot3 Burger robot and ROS 2 Humble.

The robot uses LiDAR distance measurements to maintain a target separation of 0.30 m from the left wall, navigate a rectangular laboratory enclosure, and execute right-angle turns.

2. Project Objectives

Implement a Baseline PID wall-following controller.

Implement an Extended PID controller incorporating linear speed regulation.

Maintain a target left-wall distance of 0.30 m.

Detect and navigate enclosure corners.

Test the controllers in Gazebo simulation and on the physical TurtleBot3.

Record experiment data using CSV logging.

Compare tracking accuracy, velocity response, corner handling, and controller performance.

3. Hardware and Software

Component

Description

Robot

TurtleBot3 Burger

Distance sensor

2D LiDAR

Operating system

Ubuntu 22.04

Robotics framework

ROS 2 Humble

Programming language

Python

Simulation

Gazebo

Feedback

LiDAR and wheel odometry

4. Control Methods

Baseline PID Controller

The Baseline controller regulates the robot's angular velocity using left-wall distance measurements while maintaining a configured forward velocity.

Extended PID Controller

The Extended controller provides additional forward-speed regulation to improve robot motion and wall tracking.

Both methods are evaluated under comparable laboratory conditions.

5. Experimental Setup

Enclosure: Approximately 2 m × 2 m rectangular laboratory test area.

Target wall: Left wall.

Reference distance: 0.30 m.

Corner navigation: Right-angle turns.

Evaluation: Gazebo and physical TurtleBot3.

6. Data Logging and Analysis

The ROS 2 data logger records experimental information in CSV files for subsequent analysis.

The project evaluates:

Wall-distance tracking error.

Mean absolute error (MAE).

Root mean square error (RMSE).

Linear and angular velocity.

Corner navigation performance.

Baseline versus Extended PID results.

7. ROS 2 Package

The wall_follower package contains the controller, data logger, simulation launch configuration, and supporting files.

Run the Baseline controller:

ros2 run wall_follower wall_follower_node --ros-args \
-p side:=left -p mode:=baseline -p use_sim_time:=false \
-p v_max:=0.05 -p kp_w:=3.0 -p kd_w:=0.8 \
-p lookahead:=0.2 -p front_stop:=0.34

Run the Extended controller:

ros2 run wall_follower wall_follower_node --ros-args \
-p side:=left -p mode:=extended -p use_sim_time:=false \
-p v_max:=0.05 -p v_min:=0.02 \
-p kp_w:=3.0 -p kd_w:=0.8 \
-p lookahead:=0.2 -p front_stop:=0.34
