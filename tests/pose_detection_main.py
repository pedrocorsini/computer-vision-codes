from computer_vision_modules import PoseEstimation
import cv2 as cv
import time

def main():

    #capture = cv.VideoCapture('resources/Videos/fut.mp4')
    capture = cv.VideoCapture(0)
    c_time = 0
    p_time = 0

    pose = PoseEstimation()

    while True:
        success, frame = capture.read()

        frame = pose.find_pose(frame)
        lm_list = pose.get_pose(frame, draw=False)

        if len(lm_list) != 0:
            print(lm_list[14])
            cv.circle(frame, (lm_list[14][1], lm_list[14][2]), 10, (255,0,255), cv.FILLED)

        # FPS
        c_time = time.time()
        fps = 1/(c_time - p_time)
        p_time = c_time

        cv.putText(frame, f'FPS: {int(fps)}', (70,50), cv.FONT_HERSHEY_PLAIN, 3, (255,0,0), 3)
        cv.imshow('Video', frame)
    
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

if __name__ == '__main__':
    main()