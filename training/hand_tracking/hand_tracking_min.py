import cv2 as cv
import mediapipe as mp
import time

capture = cv.VideoCapture(1)

# Default values
# static_image_mode = False
# max_num_hands = 2
# min_detection_confidence = 0.5
# min_tracking_confidence 0.5

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

pTime = 0 # Previus time
cTime = 0 # Current time

while True:
    isTrue, frame = capture.read()

    frameRGB = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    results = hands.process(frameRGB)
    # print(results.multi_hand_landmarks)

    # Drawing hands detection
    if results.multi_hand_landmarks: # Verifies if is identifying hand
        for handLms in results.multi_hand_landmarks: # Loop to draw the points
            for id, lm in enumerate(handLms.landmark): 
                # print(id, lm) # Prints the ID of all points and x, y and z positions (ratial of the image)
                h, w, c = frame.shape # height, width, channels
                cx, cy = int(lm.x*w), int(lm.y*h) # cx and cy positions (based on the center)
                print(id, cx, cy)
                # The IDs are each one of the points of the hand. 0 -> bottom point, 4 -> tip of the thumb. etc.
                # if id==4:
                #     cv.circle(frame, (cx,cy), 15, (255,0,255), cv.FILLED)
            mpDraw.draw_landmarks(frame, handLms, mpHands.HAND_CONNECTIONS) # Method to draw the points of your hand and connections

    # FPS
    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime

    cv.putText(frame, str(int(fps)), (10,70), cv.FONT_HERSHEY_COMPLEX, 2, (255,0,255), 3)

    cv.imshow('Webcam', frame)

    cv.waitKey(1)