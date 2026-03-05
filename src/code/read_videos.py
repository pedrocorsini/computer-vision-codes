import cv2 as cv

# Reading videos

# capture = cv.VideoCapture('resources/Videos/dog.mp4') # video path
capture = cv.VideoCapture(0)

while True:
    isTrue, frame = capture.read() # reads the video frame by frame, returns frame and boolean if the frame was read successfully
    cv.imshow('Video', frame)   # displays the video frame by frame

    if cv.waitKey(20) & 0xFF == ord('d'): #if the user press 'd' on the keyboard, breaks the 
        break

capture.release()
cv.destroyAllWindows()
  