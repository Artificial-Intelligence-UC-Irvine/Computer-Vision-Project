import cv2
import math
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from pose_lib import draw_pose_landmarks_on_image

def find_distance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def find_angle(x1, y1, x2, y2):
    vec_len = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    if vec_len == 0 or y1 == 0:
        return 0.0
    theta = math.acos((y2 - y1) * (-y1) / (vec_len * y1))
    degree = (180 / math.pi) * theta
    return degree


def send_warning(bad_time):
    print(f"Bad posture! Duration: {round(bad_time, 1)}s")

# Landmark index constants (mediapipe.tasks PoseLandmark ordering)
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_EAR = 7
LEFT_HIP = 23

# Color
blue = (255, 127, 0)
red = (50, 50, 255)
green = (127, 255, 0)
dark_blue = (127, 20, 0)
light_green = (127, 233, 100)
yellow = (0, 255, 255)
pink = (255, 0, 255)

font = cv2.FONT_HERSHEY_SIMPLEX

# Main loop
def run_posture_app():
    good_frames = 0
    bad_frames = 0
    warning_sent = False # prevent repeated prints per bad stretch

    base_options = python.BaseOptions(model_asset_path='pose_landmarker.task')
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        output_segmentation_masks=False,
        num_poses=1,
    )
    detector = vision.PoseLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam.")
        return

    fps_meta = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_out = cv2.VideoWriter('output.mp4', fourcc, fps_meta, (width, height))

    # Main loop
    while True:
        success, image = cap.read()
        if not success:
            print("Null frames — exiting.")
            break

        fps = cap.get(cv2.CAP_PROP_FPS) or fps_meta
        h, w = image.shape[:2]

        # Mirror for natural front-camera feel
        image = cv2.flip(image, 1)

        # Convert BGR -> RGB for mediapipe
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        # Run pose detection
        results = detector.detect(mp_image)

        # Draw skeleton on the frame
        annotated_rgb = draw_pose_landmarks_on_image(mp_image.numpy_view(), results)
        image = cv2.cvtColor(annotated_rgb, cv2.COLOR_RGB2BGR)

        # Skip posture logic if no pose detected
        if not results.pose_landmarks:
            cv2.putText(image, 'No pose detected', (10, 30), font, 0.9, red, 2)
            cv2.imshow('Posture Monitor — press Q to quit', image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        lm = results.pose_landmarks[0] # first (and only) detected person

        # Extract key landmark pixel coordinates
        l_shldr_x = int(lm[LEFT_SHOULDER].x * w)
        l_shldr_y = int(lm[LEFT_SHOULDER].y * h)

        r_shldr_x = int(lm[RIGHT_SHOULDER].x * w)
        r_shldr_y = int(lm[RIGHT_SHOULDER].y * h)

        l_ear_x = int(lm[LEFT_EAR].x * w)
        l_ear_y = int(lm[LEFT_EAR].y * h)

        l_hip_x = int(lm[LEFT_HIP].x * w)
        l_hip_y = int(lm[LEFT_HIP].y * h)

        # Camera alignment check
        offset = find_distance(l_shldr_x, l_shldr_y, r_shldr_x, r_shldr_y)
        if offset < 100:
            cv2.putText(image, f'{int(offset)} Aligned', (w-300, 30), font, 0.9, green, 2)
        else:
            cv2.putText(image, f'{int(offset)} Not Aligned', (w-300, 30), font, 0.9, red, 2)

        # Angles
        neck_inclination = find_angle(l_shldr_x, l_shldr_y, l_ear_x, l_ear_y)
        torso_inclination = find_angle(l_hip_x, l_hip_y, l_shldr_x, l_shldr_y)

        # Landmark dots (reference circles)
        cv2.circle(image, (l_shldr_x, l_shldr_y), 7, yellow, -1)
        cv2.circle(image, (l_ear_x, l_ear_y), 7, yellow, -1)
        cv2.circle(image, (l_shldr_x, l_shldr_y - 100), 7, yellow, -1) # vertical ref
        cv2.circle(image, (r_shldr_x, r_shldr_y), 7, pink, -1)
        cv2.circle(image, (l_hip_x, l_hip_y), 7, yellow, -1)
        cv2.circle(image, (l_hip_x, l_hip_y - 100), 7, yellow, -1) # vertical ref

        angle_text = f'Neck: {int(neck_inclination)}; Torso: {int(torso_inclination)}'

        # Posture classification
        if neck_inclination < 40 and torso_inclination < 10:
            bad_frames = 0
            good_frames += 1
            warning_sent = False
            colour = light_green
            line_colour = green
        else:
            good_frames = 0
            bad_frames += 1
            colour = red
            line_colour = red

        cv2.putText(image, angle_text, (10, 30), font, 0.9, colour, 2)
        cv2.putText(image, str(int(neck_inclination)), (l_shldr_x + 10, l_shldr_y), font, 0.9, colour, 2)
        cv2.putText(image, str(int(torso_inclination)), (l_hip_x + 10, l_hip_y), font, 0.9, colour, 2)

        # Skeleton lines
        cv2.line(image, (l_shldr_x, l_shldr_y), (l_ear_x, l_ear_y), line_colour, 4)
        cv2.line(image, (l_shldr_x, l_shldr_y), (l_shldr_x, l_shldr_y - 100), line_colour, 4)
        cv2.line(image, (l_hip_x, l_hip_y), (l_shldr_x, l_shldr_y), line_colour, 4)
        cv2.line(image, (l_hip_x, l_hip_y), (l_hip_x, l_hip_y - 100), line_colour, 4)

        # Timing
        good_time = (1 / fps) * good_frames
        bad_time = (1 / fps) * bad_frames

        if good_time > 0:
            cv2.putText(image, f'Good Posture Time : {round(good_time, 1)}s', (10, h - 20), font, 0.9, green, 2)
        else:
            cv2.putText(image, f'Bad Posture Time : {round(bad_time, 1)}s', (10, h - 20), font, 0.9, red, 2)

        # 3 min bad posture
        if bad_time > 180 and not warning_sent:
            send_warning(bad_time)
            warning_sent = True

        # Controls display
        cv2.putText(image, 'Q = Quit', (10, h - 50), font, 0.55, (180, 180, 180), 1)

        video_out.write(image)
        cv2.imshow('Posture Detector', image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    video_out.release()
    detector.close()
    cv2.destroyAllWindows()
    for _ in range(5):
        cv2.waitKey(1)
    print("Camera released and windows closed successfully.")

if __name__ == '__main__':
    run_posture_app()