import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
from library import draw_landmarks_on_image

MARGIN = 10
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54)  # vibrant green

# def draw_landmarks_on_image(rgb_image, detection_result):
#     hand_landmarks_list = detection_result.hand_landmarks
#     handedness_list = detection_result.handedness
#     annotated_image = np.copy(rgb_image)
#     height, width, _ = annotated_image.shape

#     for idx in range(len(hand_landmarks_list)):
#         hand_landmarks = hand_landmarks_list[idx]
#         handedness = handedness_list[idx]

#         # Draw connections (bones) first so dots appear on top
#         for connection in mp.tasks.vision.HandLandmarksConnections.HAND_CONNECTIONS:
#             start = hand_landmarks[connection.start]
#             end = hand_landmarks[connection.end]
#             start_pt = (int(start.x * width), int(start.y * height))
#             end_pt = (int(end.x * width),   int(end.y * height))
#             cv2.line(annotated_image, start_pt, end_pt, (255, 255, 255), 2)

#         # Draw landmark dots
#         for lm in hand_landmarks:
#             cx = int(lm.x * width)
#             cy = int(lm.y * height)
#             cv2.circle(annotated_image, (cx, cy), 5, (0, 255, 0), -1)

#         # Draw Left / Right label above the hand
#         x_coords = [lm.x for lm in hand_landmarks]
#         y_coords = [lm.y for lm in hand_landmarks]
#         text_x = int(min(x_coords) * width)
#         text_y = int(min(y_coords) * height) - MARGIN

#         cv2.putText(annotated_image, f"{handedness[0].category_name}",
#                     (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
#                     FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)

#     return annotated_image


base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=2)
detector = vision.HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Could not open webcam.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to read from camera.")
        break

    frame = cv2.flip(frame, 1)

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    detection_result = detector.detect(mp_image)

    annotated = draw_landmarks_on_image(mp_image.numpy_view(), detection_result)

    cv2.imshow("Hand Tracking", cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
detector.close()
cv2.destroyAllWindows()