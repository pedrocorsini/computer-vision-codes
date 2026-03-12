from computer_vision_modules import HandDetector
import cv2 as cv
import time

def main():
    p_time = 0 # Previus time
    c_time = 0 # Current time

    capture = cv.VideoCapture(0)
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
