import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import cv2
import mediapipe as mp


class FollowMe(Node):
    def __init__(self):
        super().__init__('follow_me')
        self.bridge = CvBridge()

        # Subscribing to camera feed
        self.image_sub = self.create_subscription(Image, 'camera/image_raw', self.image_callback, 10)

        # Publisher for movement
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        # Debug image publisher
        self.debug_pub = self.create_publisher(Image, '/mediapipe_followme/image', 10)

        # MediaPipe pose
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(min_detection_confidence=0.5, model_complexity=1)

        self.image_width = 640
        self.center_x = self.image_width // 2  # 160
        self.deadzone_px = 20  # How close to the center before moving forward
        self.prev_linear_x = 0.0

    def image_callback(self, msg):
        twist = Twist()

        # Convert ROS image to OpenCV format
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        image_width = cv_image.shape[1]

        # Process image
        results = self.pose.process(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))

        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                cv_image, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS
            )

            landmarks = results.pose_landmarks.landmark
            ankle = landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE]

            ############################
            # Print left ankle coordinates
            ############################

            ############################
            # --- Landmark Processing and Visualization ---
            # Extract pixel coordinates of the left ankle.
            # Note: MediaPipe normalized coordinates (0.0 to 1.0) need to be scaled by image dimensions.
            # It's safer to use cv_image.shape[1] for width to ensure it matches the current frame.   
	    ############################

            ############################
            # Log ankle coordinates for debugging.
            ############################

            ############################
	    # Draw a blue circle around the detected left ankle on the image.
	    # Parameters: image, center_coordinates, radius, color (BGR), thickness
	    ############################


            # --- Robot Control Logic ---
            # Calculate the horizontal offset of the ankle from the image center.
            # A positive offset means the ankle is to the right of the center.
            offset_from_center = ankle_x_pixel - (cv_image.shape[1] // 2)


            # Check if the ankle landmark's visibility confidence is below a threshold.
            # If visibility is low, it indicates an unreliable detection.
            if ankle.visibility < 0.5:
                self.get_logger().warn('Ankle visibility too low (< 0.5). Stopping robot.')
                twist.linear.x = 0.0
                twist.angular.z = 0.0
            
            ############################
            # Angular control 
            # Check if the ankle is outside the acceptable horizontal tolerance zone.
            # This indicates the robot needs to rotate to re-center the ankle.
            # P-controller (Greater error = greater turn speed)
            ############################
           
            ############################
            # If the ankle is within the tolerance zone, it's considered centered.
	    # Ankle is centered, move forward at a constant linear speed.
	    ############################
                

        else:
            self.get_logger().info('No landmarks. Stopping.')
            twist.linear.x = 0.0
            twist.angular.z = 0.0

        self.cmd_pub.publish(twist)

        # --- Debug Image Publishing (Moved and Consolidated) ---
            # This block is now outside the 'if results.pose_landmarks' to ensure a debug image
            # is always published, even if no landmarks are detected (it will just be the raw image then).
            # The previous debug_pub.publish(debug_msg) inside the 'if' block is removed to avoid redundancy.
        try:
            # Convert the (potentially annotated) OpenCV image back to a ROS Image message.
            debug_msg = self.bridge.cv2_to_imgmsg(cv_image, encoding='bgr8')
            self.debug_pub.publish(debug_msg)
        except Exception as e:
            self.get_logger().error(f"Failed to publish debug image: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = FollowMe()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()




