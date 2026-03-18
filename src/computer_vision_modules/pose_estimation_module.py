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

class PoseEstimation():
    def __init__(self, mode=False, complexity=1, smooth_lmks=True, segmentation=False, smooth_seg=True, detection_confidence=0.5, tracking_confidence=0.5):
        self.mode = mode
        self.complexity = complexity
        self.smooth_lmks = smooth_lmks
        self.segmentation = segmentation
        self.smooth_seg = smooth_seg
        self.detection_confidence = detection_confidence
        self.tracking_confidence = tracking_confidence

        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(static_image_mode = self.mode,
                                 model_complexity = self.complexity,
                                 smooth_landmarks = self.smooth_lmks,
                                 enable_segmentation = self.segmentation,
                                 smooth_segmentation = self.smooth_seg,
                                 min_detection_confidence = self.detection_confidence,
                                 min_tracking_confidence = self.tracking_confidence
                                 )
        self.mp_draw = mp.solutions.drawing_utils

    def find_pose(self, frame, draw=True):
        frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        self.results = self.pose.process(frame_rgb) # Process the pose estimation
        # Drawing the results
        if self.results.pose_landmarks:
            if draw:
                self.mp_draw.draw_landmarks(frame, self.results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
        return frame 
    
    def get_pose(self, frame, draw=True):
        lm_list=[]
        if self.results.pose_landmarks:
    
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = frame.shape
                #print(id, lm)
                cx, cy = int(lm.x*w), int(lm.y*h)
                lm_list.append([id, cx, cy])
                if draw:
                    cv.circle(frame, (cx, cy), 3, (255,0,255), cv.FILLED) # Just to check if the pixel values (cx, cy) of each landmark is correct
        return lm_list

def main():

    capture = cv.VideoCapture('resources/videos/fut.mp4')
    # capture = cv.VideoCapture(0)
    c_time = 0
    p_time = 0

    pose = PoseEstimation()

    while True:
        success, frame = capture.read()

        frame = pose.find_pose(frame)
        lm_list = pose.get_pose(frame, draw=False)

        if len(lm_list) != 0:
            print(lm_list[14])

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