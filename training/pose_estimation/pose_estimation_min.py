import cv2 as cv
import mediapipe as mp
import time

# Uses mediapipe solutions (deprecated)

# Pose Estimation Default Values
#    static_image_mode=False
#    model_complexity=1
#    smooth_landmarks=True
#    enable_segmentation=False
#    smooth_segmentation=True
#    min_detection_confidence=0.5
#    min_tracking_confidence=0.5

mp_draw = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

capture = cv.VideoCapture('resources/Videos/fut.mp4')
# capture = cv.VideoCapture(0)

c_time = 0
p_time = 0

while True:
    success, frame = capture.read()
    frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    results = pose.process(frame_rgb) # Process the pose estimation
    #print(results.pose_landmarks)

    # Check if the detection works
    # Only works for 1 person
    if results.pose_landmarks:
        mp_draw.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        for id, lm in enumerate(results.pose_landmarks.landmark):
            h, w, c = frame.shape
            print(id, lm)
            cx, cy = int(lm.x*w), int(lm.y*h)
            #cv.circle(frame, (cx, cy), 5, (255,0,255), cv.FILLED) # Just to check if the pixel values (cx, cy) of each landmark is correct

    
    c_time = time.time()
    fps = 1/(c_time - p_time)
    p_time = c_time

    # width = 1365
    # height = 720
    # dimensions = (width, height)

    # frame_resized = cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)

    cv.putText(frame, f'FPS: {int(fps)}', (70,50), cv.FONT_HERSHEY_PLAIN, 3, (255,0,0), 3)
    cv.imshow('Video', frame)
    # print(frame_resized.shape)


    if cv.waitKey(1) & 0xFF == ord('q'):
        break