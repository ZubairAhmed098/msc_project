import pyrealsense2 as rs
import cv2
import numpy as np
import mediapipe as mp

# Initialize RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
pipeline.start(config)

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# Function to apply custom depth colormap
def apply_custom_colormap(depth_frame):
    depth_frame = np.clip(depth_frame, 0, 4000)  # Clip to avoid extreme values
    depth_norm = cv2.normalize(depth_frame, None, 0, 255, cv2.NORM_MINMAX)  # Normalize
    depth_norm = np.uint8(depth_norm)  # Convert to 8-bit
    
    # Apply COLORMAP_JET (blue -> green -> yellow -> red)
    depth_colormap = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)
    return depth_colormap

try:
    while True:
        # Get frames
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        if not depth_frame or not color_frame:
            continue

        # Convert frames to numpy arrays
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())

        # Apply custom depth colormap
        depth_colormap = apply_custom_colormap(depth_image)

        # Convert color image to RGB for MediaPipe
        rgb_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

        # Process with MediaPipe Pose
        results = pose.process(rgb_image)

        # Draw skeleton if detected
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(color_image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Combine images side by side
        combined_image = np.hstack((color_image, depth_colormap))

        # Display
        cv2.imshow("Color (Skeleton) | Depth (Distance)", combined_image)

        # Exit on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()

