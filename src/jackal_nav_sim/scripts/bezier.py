#!/usr/bin/env python3

import rospy
import numpy as np
from geometry_msgs.msg import PoseStamped, Twist, Point, Quaternion
from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker

class BezierVisualizer:
    def __init__(self):
        rospy.init_node('bezier_visualizer')
        
        # Poses
        self.robot_pose = None
        self.goal_pose = None
        self.bezier_points = []
        
        # Publishers
        self.marker_pub = rospy.Publisher('/bezier_curve', Marker, queue_size=10)
        self.lookahead_marker_pub = rospy.Publisher('/lookahead_point', Marker, queue_size=10)
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        #get Odom and goal with 2D nav
       # rospy.Subscriber('/jackal_velocity_controller/odom', Odometry, self.odom_callback)
        rospy.Subscriber('/odometry/filtered', Odometry, self.odom_callback)

        rospy.Subscriber('/move_base_simple/goal', PoseStamped, self.goal_callback)
        # Initialize marker
        self.marker = Marker()
        self.marker.header.frame_id = "odom"
        self.marker.type = Marker.LINE_STRIP
        self.marker.action = Marker.ADD
        self.marker.scale.x = 0.1
        self.marker.color.a = 1.0
        self.marker.color.g = 1.0
        #Lookahead
        self.lookahead_marker = Marker()
        self.lookahead_marker.header.frame_id = "odom"
        self.lookahead_marker.type = Marker.SPHERE  # My lookahead Dot
        self.lookahead_marker.action = Marker.ADD
        self.lookahead_marker.scale.x = 0.2  # Size of the sphere
        self.lookahead_marker.scale.y = 0.2
        self.lookahead_marker.scale.z = 0.2
        self.lookahead_marker.color.r = 1.0  
        self.lookahead_marker.color.g = 0.2  
        self.lookahead_marker.color.b = 1.0  
        self.lookahead_marker.color.a = 1.0  # Opacity
        
        self.rate = rospy.Rate(10) 
        rospy.loginfo("Bezier visualizer node started")

    def odom_callback(self, msg):
        """Callback for robot odometry updates"""
        self.robot_pose = msg.pose.pose
        self.pure_pursuit()

    def goal_callback(self, msg):
        """Callback for new goal poses"""
        self.goal_pose = msg.pose
        self.update_bezier()

    def get_yaw(self, quaternion):
        """Extract yaw angle from quaternion"""
        siny_cosp = 2.0 * (quaternion.w * quaternion.z + quaternion.x * quaternion.y)
        cosy_cosp = 1.0 - 2.0 * (quaternion.y**2 + quaternion.z**2)
        return np.arctan2(siny_cosp, cosy_cosp)


#TODO Need to take the bezier function and create it's own class and ROS node
    def calculate_bezier_points(self, start, end, num_points=50):
        """Calculate points along a cubic Bezier curve with improved control point placement."""
        print("Calculating Curve...")
        dist = np.sqrt((end.position.x - start.position.x)**2 + 
                    (end.position.y - start.position.y)**2)
        #Checks
        if dist < 0.1:
            return [start.position]
        
        # Control points for our bezier curves
        control_distance = dist * 0.5  
        
        #Yaws
        start_yaw = self.get_yaw(start.orientation)
        end_yaw = self.get_yaw(end.orientation)
        
        # Midpoints for control points
        midpoint_x = (start.position.x + end.position.x) / 2
        midpoint_y = (start.position.y + end.position.y) / 2

        # Offset control points, basically the "sides"
        control1_x = midpoint_x + control_distance * 0.5 * np.cos(start_yaw)
        control1_y = midpoint_y + control_distance * 0.5 * np.sin(start_yaw)
        control2_x = midpoint_x - control_distance * 0.5 * np.cos(end_yaw)
        control2_y = midpoint_y - control_distance * 0.5 * np.sin(end_yaw)
        # Generate points 
        points = []
        for t in np.linspace(0, 1, num_points):
            # Cubic formula
            x = (1-t)**3 * start.position.x + \
                3*(1-t)**2*t * control1_x + \
                3*(1-t)*t**2 * control2_x + \
                t**3 * end.position.x
            y = (1-t)**3 * start.position.y + \
                3*(1-t)**2*t * control1_y + \
                3*(1-t)*t**2 * control2_y + \
                t**3 * end.position.y
            
            point = Point()
            point.x = x
            point.y = y
            point.z = 0.1
            points.append(point)
        print("Calculated said curve!!!!")
        return points

    def update_bezier(self):
        """Update Bezier curve visualization"""
        if self.robot_pose is None or self.goal_pose is None:
            return
        
        self.bezier_points = self.calculate_bezier_points(self.robot_pose, self.goal_pose)
        self.marker.header.stamp = rospy.Time.now()
        self.marker.points = self.bezier_points
        self.marker_pub.publish(self.marker)

    def pure_pursuit(self, lookahead_distance=0.02):
        if not self.bezier_points or self.robot_pose is None or self.goal_pose is None:
            return 
        
        robot_x = self.robot_pose.position.x
        robot_y = self.robot_pose.position.y

        # First find the closest point on the curve
        closest_idx = 0
        min_dist = float('inf') #not too sure seems right
        for i, point in enumerate(self.bezier_points):
            dist = np.hypot(point.x - robot_x, point.y - robot_y)
            if dist < min_dist:
                min_dist = dist
                closest_idx = i

        # Check if we've reached the goal
        goal_distance = np.hypot(self.goal_pose.position.x - robot_x, 
                            self.goal_pose.position.y - robot_y) #hypot triangle
        goal_tolerance = 0.1

        if goal_distance < goal_tolerance:
            stop_cmd = Twist()
            self.cmd_vel_pub.publish(stop_cmd)
            print("Goal reached!")
            return
        base_lookahead = 0.3
        min_lookahead = 0.2
        
        # Find lookahead point on curve ahead of closest point
        lookahead_point = None
        lookahead_idx = closest_idx
        search_window = min(len(self.bezier_points) - closest_idx, 70) 
        
        for i in range(closest_idx, closest_idx + search_window):
            if i >= len(self.bezier_points):
                break
                
            point = self.bezier_points[i]
            distance = np.hypot(point.x - robot_x, point.y - robot_y)
            
            if distance >= base_lookahead:
                lookahead_point = point
                lookahead_idx = i
                break
        
        # More checks
        if lookahead_point is None:
            lookahead_point = self.bezier_points[-1]
            lookahead_idx = len(self.bezier_points) - 1

        # Calculate path tangent at lookahead point
        if lookahead_idx < len(self.bezier_points) - 1:
            next_point = self.bezier_points[lookahead_idx + 1]
            path_tangent = np.arctan2(next_point.y - lookahead_point.y,
                                    next_point.x - lookahead_point.x)
        else:
            path_tangent = self.get_yaw(self.goal_pose.orientation)
        dx = lookahead_point.x - robot_x
        dy = lookahead_point.y - robot_y

        target_yaw = np.arctan2(dy, dx)
        current_yaw = self.get_yaw(self.robot_pose.orientation)

        path_weight = min(1.0, goal_distance / 1.2) 
        target_yaw = path_weight * path_tangent + (1 - path_weight) * target_yaw
        
        angular_error = np.arctan2(np.sin(target_yaw - current_yaw), 
                                np.cos(target_yaw - current_yaw))

        # Calculate tracking error (cross track error)
        cross_track_error = min_dist

        # Adjust velocities based on curve following
        curvature = abs(angular_error) / base_lookahead
        self.lookahead_marker.header.stamp = rospy.Time.now()
        self.lookahead_marker.pose.position = lookahead_point
        self.lookahead_marker.pose.orientation.w = 1.0
        self.lookahead_marker_pub.publish(self.lookahead_marker)
        # Some control calculations
        base_velocity = 0.5  # Max speeeeed
        curve_factor = 1.0 / (1.0 + 2.0 * curvature)  # Slow down on curves
        tracking_factor = 1.0 / (1.0 + 0.5 * cross_track_error)  # Slow down when off path
        linear_velocity = base_velocity * curve_factor * tracking_factor
        angular_velocity = 1.5 * angular_error + 0.5 * cross_track_error * np.sign(angular_error)

        cmd = Twist()
        cmd.linear.x = np.clip(linear_velocity, 0, 2.0)
        cmd.angular.z = np.clip(angular_velocity, -2.0, 2.0)
        self.cmd_vel_pub.publish(cmd)
        
        # Debugs
        print(f"Cross track error: {cross_track_error:.2f}m")
        print(f"Curvature: {curvature:.2f}")
        print(f"Velocities - linear: {cmd.linear.x:.2f}, angular: {cmd.angular.z:.2f}")

    def run(self):
        """Main run loop"""
        while not rospy.is_shutdown():
            self.rate.sleep()

if __name__ == '__main__':
    try:
        visualizer = BezierVisualizer()
        visualizer.run()
    except rospy.ROSInterruptException:
        pass