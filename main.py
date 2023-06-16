import cv2, mediapipe as mp
import math

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
cap = cv2.VideoCapture("squat.mp4")
push_up = None

def angle(point1, point2, point3):
    """ Calculate angle between two lines """
    point1 = (point1.x, point1.y)
    point2 = (point2.x, point2.y)
    point3 = (point3.x, point3.y)

    if(point1==(0,0) or point2==(0,0) or point3==(0,0)):
        return 0
    numerator = point2[1] * (point1[0] - point3[0]) + point1[1] * \
                (point3[0] - point2[0]) + point3[1] * (point2[0] - point1[0])
    denominator = (point2[0] - point1[0]) * (point1[0] - point3[0]) + \
                (point2[1] - point1[1]) * (point1[1] - point3[1])
    try:
        ang = math.atan(numerator/denominator)
        ang = ang * 180 / math.pi
        if ang < 0:
            ang = 180 + ang
        return ang
    except:
        return 90.0

def angle_of_singleline(point1, point2):

    point1 = (point1.x, point1.y)
    point2 = (point2.x, point2.y)

    """ Calculate angle of a single line """
    x_diff = point2[0] - point1[0]
    y_diff = point2[1] - point1[1]
    return math.degrees(math.atan2(y_diff, x_diff))

with mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7) as pose:
    while cap.isOpened():
        ret, frame = cap.read()

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)

        left_wrist = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST]
        right_wrist = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]
        left_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]

        # print("Left: wrist ",left_wrist.y," shoulder", left_shoulder.y)
        # print("Right: wrist ",right_wrist.y," shoulder", right_shoulder.y)

        left_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        left_hip = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]
        right_hip = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP]
        left_ankle = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE]
        right_ankle = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE]
        left_elbow = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW]
        right_elbow = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW]
        left_wrist = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST]
        right_wrist = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]

        wrist_elbow_shoulder_angle = angle(left_wrist, left_elbow, left_shoulder)
        elbow_shoulder_ankle_angle = angle(left_elbow, left_shoulder, left_ankle)
        shoulder_hip_ankle_angle = angle(left_shoulder, left_hip, left_ankle)
        # elbow_wrist_angle_horizontal = math.degrees(math.atan2(right_wrist.x - right_elbow.x, right_wrist.y - right_elbow.y))


        # print("Shoulder-hip angle: ", shoulder_hip_angle)
        print("wrist_elbow_shoulder angle: ", wrist_elbow_shoulder_angle)
        print("elbow-shoulder-ankle angle: ", elbow_shoulder_ankle_angle)
        print("shoulder-hip-ankle angle: ", shoulder_hip_ankle_angle)
        print("\n")

        # Check if push-up position is detected based on the conditions
        if(wrist_elbow_shoulder_angle < 180) and (elbow_shoulder_ankle_angle<90 or elbow_shoulder_ankle_angle>170) and (shoulder_hip_ankle_angle<180):
            print("Pushup detected")
            push_up = True

        else:
            push_up = False

        # push_up  = True
        # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # print(results)
        if push_up:
            cv2.putText(frame, 'Push-up Detected', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(frame, 'No Push-up Detected', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS, mp_drawing.DrawingSpec(color=(100,100,100), thickness=1, circle_radius=3), mp_drawing.DrawingSpec(color=(0,0,0), thickness=1, circle_radius=3))


        cv2.imshow("Mediapipe feed", frame)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()