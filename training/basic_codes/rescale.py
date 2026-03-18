import cv2 as cv

# Function to rescale frames
def rescaleFrame(frame, scale=0.75):
    # Works for images, videos and live videos
    width = int(frame.shape[1] * scale) # shape[1] -> width
    height = int(frame.shape[0] * scale) # shape[0] -> height
    dimensions = (width, height)

    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)

def changeRes(width, height):
    # Only works for live videos 
    capture.set(3,width)
    capture.set(4,height)

img = cv.imread('resources/Photos/cat.jpg')
resized_image = rescaleFrame(img)
cv.imshow('Image', img)
cv.imshow('Resized Image', resized_image)

capture = cv.VideoCapture('resources/Videos/kitten.mp4')

while True:
    isTrue, frame = capture.read() # reads the video frame by frame, returns frame and boolean if the frame was read successfully

    frame_resized = rescaleFrame(frame, scale=.2)
    
    cv.imshow('Video', frame)   # displays the video frame by frame
    cv.imshow('Video Resized', frame_resized) 

    if cv.waitKey(20) & 0xFF == ord('d'): #if the user press 'd' on the keyboard, breaks the loop
        break

capture.release()
cv.destroyAllWindows()