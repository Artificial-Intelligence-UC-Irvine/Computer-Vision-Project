import cv2
import mediapipe as mp
import numpy as np

# Pose landmark connections from mediapipe tasks
POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7),
    (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10),
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (24, 26), (26, 28),
    (27, 29), (29, 31), (28, 30), (30, 32),
]

def draw_pose_landmarks_on_image(rgb_image, detection_result):
    """
    Draws pose landmark dots and bone connections onto a copy of rgb_image.
    Mirrors the structure of draw_landmarks_on_image in library.py.
    """
    annotated_image = np.copy(rgb_image)
    height, width, _ = annotated_image.shape

    if not detection_result.pose_landmarks:
        return annotated_image

    for pose_landmarks in detection_result.pose_landmarks:
        # Draw connections (bones) first so dots appear on top
        for start_idx, end_idx in POSE_CONNECTIONS:
            start = pose_landmarks[start_idx]
            end = pose_landmarks[end_idx]
            start_pt = (int(start.x * width), int(start.y * height))
            end_pt = (int(end.x * width), int(end.y * height))
            cv2.line(annotated_image, start_pt, end_pt, (255, 255, 255), 2)

        # Draw landmark dots
        for lm in pose_landmarks:
            cx = int(lm.x * width)
            cy = int(lm.y * height)
            cv2.circle(annotated_image, (cx, cy), 5, (0, 255, 0), -1)

    return annotated_image