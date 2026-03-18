from computer_vision_modules import HandDetector
import cv2 as cv
import time
import math

def main():
    p_time = 0 # Previus time
    c_time = 0 # Current time

    capture = cv.VideoCapture(0)
    detector = HandDetector()
    
    while True:
        success, frame = capture.read()
        if not success:
            break
        frame = detector.find_hands(frame)
        lm_list = detector.find_position(frame, draw=False)
        dist_pixels = 0
        # FOCAL_LENGTH = 1200
        # DISTANCE_HAND_CAMERA = 50 

        if len(lm_list) != 0:
            print(lm_list[4])
            print(lm_list[8])
            cv.circle(frame, (lm_list[4][1], lm_list[4][2]), 5, (0,255,0), -1)
            cv.circle(frame, (lm_list[8][1], lm_list[8][2]), 5, (0,255,0), -1)
            cv.line(frame, (lm_list[4][1], lm_list[4][2]), (lm_list[8][1], lm_list[8][2]), (255,0,0), 3)
           
            dist_pixels = math.hypot(lm_list[8][1] - lm_list[4][1], lm_list[8][2] - lm_list[4][2])
            if dist_pixels != 0:
                # if dist_pixels > 0:
                #     dist_cm = (dist_pixels * DISTANCE_HAND_CAMERA) / FOCAL_LENGTH
                cv.putText(frame, f'Pixels Distance: {int(dist_pixels)}', (40, 400), cv.FONT_HERSHEY_PLAIN, 2, (0,0,255), 3)

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
