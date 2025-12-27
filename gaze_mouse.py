import cv2
import mediapipe as mp
import numpy as np
import pyautogui as py
import time

py.FAILSAFE = True
MOVE_SPEED = 25
GAZE_HOLD_TIME = 3.0

LEFT_THRESHOLD = 0.35
RIGHT_THRESHOLD = 0.65
UP_THRESHOLD = 0.25
DOWN_THRESHOLD = 0.75
VERTICAL_CENTER = 0.55
VERTICAL_DEADZONE = 0.12
DOWN_EAR_THRESHOLD = 0.22


EAR_BLINK_THRESH = 0.18

screen_w, screen_h = py.size()

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

cap = cv2.VideoCapture(0)

blink_counter = 0
blink_start_time = 0
lasr_action_time = time.time()
gaze_hold_start = None

def eye_aspect_ratio(eye_points):
    vertical1 = np.linalg.norm(eye_points[1] - eye_points[5])
    vertical2 = np.linalg.norm(eye_points[2] - eye_points[4])
    horizontal = np.linalg.norm(eye_points[0] - eye_points[3])
    return (vertical1 + vertical2) / (2.0*horizontal) #Detects eye blinks

def get_gaze_ratio(eye_points, iris_points):
    eye_left_x = eye_points[0][0]
    eye_right_x = eye_points[3][0]
    iris_x = iris_points[0]
    return (iris_x - eye_left_x) / (eye_right_x - eye_left_x) #Gaze_ratio = iris distance from left/ eye width

def get_vertical_ratio(eye_pts, iris):
    top_y = eye_pts[1][1]
    bottom_y = eye_pts[4][1]
    iris_y = iris[1]
    ratio = (iris_y - top_y)/(bottom_y - top_y)
    return 1.0 - ratio

vertical_center = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1) #mirror imaging
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) #Converts to RGB cuz Mediapipe trained on RGB images

    results = face_mesh.process(rgb)

    if results.multi_face_landmarks is not None:
        landmarks = results.multi_face_landmarks[0].landmark

        left_eye = [33, 160, 158, 133, 153, 144]
        right_eye = [362, 385, 387, 263, 373, 380]

        left_eye_pts = np.array(
            [(int(landmarks[i].x*w), int(landmarks[i].y*h)) for i in left_eye]
        )
        right_eye_pts = np.array(
            [(int(landmarks[i].x*w), int(landmarks[i].y*h)) for i in right_eye]
        )

        left_iris = landmarks[468]
        iris_x = int(left_iris.x*w)
        iris_y = int(left_iris.y*h)

        #Gaze movemont
        gaze_ratio = get_gaze_ratio(left_eye_pts, (iris_x, iris_y))
        vertical_ratio = get_vertical_ratio(left_eye_pts, (iris_x, iris_y))

        #Blink Detection
        left_ear = eye_aspect_ratio(left_eye_pts)
        right_ear = eye_aspect_ratio(right_eye_pts)
        ear = (left_ear+right_ear)/2

        print(f"EAR={ear:.2f} | vertical_ratio={vertical_ratio:.2f}")

        if vertical_center is None:
            vertical_center = vertical_ratio
            print("caliberated vertical center:", vertical_center)

        moved = False

        if gaze_ratio < LEFT_THRESHOLD:
            py.moveRel(-MOVE_SPEED, 0)
            moved = True
        elif gaze_ratio > RIGHT_THRESHOLD:
            py.moveRel(MOVE_SPEED, 0)
            moved = True
        else:
            vertical_center = 0.5

            if ear < DOWN_EAR_THRESHOLD and ear > EAR_BLINK_THRESH:
                py.moveRel(0, MOVE_SPEED)
                moved = True
            elif vertical_ratio > VERTICAL_CENTER + 0.18:
                py.moveRel(0, -MOVE_SPEED)
                moved = True
            else:
                pass

        #Center Gaze
        if not moved:
            if gaze_hold_start is None:
                gaze_hold_start = time.time()
            elif time.time() - gaze_hold_start >= GAZE_HOLD_TIME:
                py.click()
                gaze_hold_start = None
        else:
            gaze_hold_start = None


        if ear < EAR_BLINK_THRESH and not moved:
            if blink_counter == 0:
                blink_start_time = time.time() 
            blink_counter += 1
        else:
            blink_duration = time.time() - blink_start_time

            if blink_counter >= 2 and blink_counter> 4:
                py.click()
            elif blink_counter >= 4:
                py.rightClick()

            blink_counter = 0


    cv2.imshow("Gaze Controlled Mouse", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()