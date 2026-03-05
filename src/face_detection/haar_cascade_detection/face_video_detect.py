import cv2 as cv

capture = cv.VideoCapture(0)

haar_cascade = cv.CascadeClassifier('src/face_detection/haar_cascade_detection/haar_face.xml')

while True:
    isTrue, frame = capture.read() # reads the video frame by frame, returns frame and boolean if the frame was read successfully
    gray_video = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    faces_rect = haar_cascade.detectMultiScale(gray_video, scaleFactor=1.1, minNeighbors=3)
    for (x,y,w,h) in faces_rect:
        cv.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), thickness=2)

    #flipped_video = cv.flip(frame, 1)

    cv.imshow('Video', frame)   # displays the video frame by frame

    if cv.waitKey(20) & 0xFF == ord('d'): #if the user press 'd' on the keyboard, breaks the 
        break

capture.release()
cv.destroyAllWindows()