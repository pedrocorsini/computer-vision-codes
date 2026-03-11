import cv2 as cv
import mediapipe as mp
import time

# Default values
# static_image_mode = False
# max_num_hands = 2
# min_detection_confidence = 0.5
# min_tracking_confidence 0.5

class HandDetector():
    def __init__(self, mode=False, max_hands = 2, detection_confidence = 0.5, track_confidence = 0.5):
        self.mode = mode
        self.max_hands = max_hands
        self.detection_confidence = detection_confidence
        self.track_confidence = track_confidence

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode=self.mode, max_num_hands=self.max_hands, 
                                        min_detection_confidence=self.detection_confidence, min_tracking_confidence=self.track_confidence)
        self.mp_draw = mp.solutions.drawing_utils

    def find_hands(self, frame, draw=True):
        frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        self.results = self.hands.process(frame_rgb)
        # print(results.multi_hand_landmarks)

        # Drawing hands detection
        if self.results.multi_hand_landmarks: # Verifies if is identifying hand
            for hand_lms in self.results.multi_hand_landmarks: # Loop to draw the points
                if draw:
                    self.mp_draw.draw_landmarks(frame, hand_lms, self.mp_hands.HAND_CONNECTIONS) # Method to draw the points of your hand and connections
        return frame


    def find_position(self, frame, hand_number=0, draw=True):
        lm_list = []
        if self.results.multi_hand_landmarks:
            my_hand = self.results.multi_hand_landmarks[hand_number]
            for id, lm in enumerate(my_hand.landmark): 
                # print(id, lm) # Prints the ID of all points and x, y and z positions (ratial of the image)
                h, w, c = frame.shape # height, width, channels
                cx, cy = int(lm.x*w), int(lm.y*h) # cx and cy positions (based on the center)
                # print(id, cx, cy)
                lm_list.append([id, cx, cy])
                # The IDs are each one of the points of the hand. 0 -> bottom point, 4 -> tip of the thumb. etc.
                if draw:
                    cv.circle(frame, (cx,cy), 7, (255,0,0), cv.FILLED)

        return lm_list

def main():
    p_time = 0 # Previus time
    c_time = 0 # Current time

    capture = cv.VideoCapture(1)

    detector = HandDetector()
    
    while True:
        success, frame = capture.read()
        frame = detector.find_hands(frame)
        lm_list = detector.find_position(frame, draw=False)
        if len(lm_list) != 0:
            print(lm_list[4])

        # FPS
        c_time = time.time()
        fps = 1/(c_time - p_time)
        p_time = c_time

        cv.putText(frame, f'FPS: {int(fps)}', (10,70), cv.FONT_HERSHEY_PLAIN, 2, (255,0,0), 3)
        cv.imshow('Webcam', frame)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break
        
    capture.release()
    cv.destroyAllWindows()

if __name__ == '__main__':
    main()
