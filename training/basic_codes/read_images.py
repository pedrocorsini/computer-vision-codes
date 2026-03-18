import cv2 as cv

# Reading images

img = cv.imread('resources/Photos/cat.jpg')

# Displays the picture on the PC

cv.imshow('Cat', img)

# Waits for user input to stop the code
cv.waitKey(0)
  