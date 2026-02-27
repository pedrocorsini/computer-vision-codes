import cv2 as cv
import numpy as np

blank = np.zeros((500,500,3), dtype='uint8') # (width, height, color channels)
cv.imshow('Blank', blank)

# 1. Paint the image
blank[200:300, 300:400] = 255,0,0 # blue, green, red
cv.imshow('Blue', blank)
 
# 2. Draw a rectangle
# draws a rectangle cv.rectangle(image, pt1, pt2, color, thickness)
cv.rectangle(blank, (0,0), (blank.shape[1]//2, blank.shape[0]//2), (0,255,0), thickness=-1) 
cv.imshow('Rectangle', blank)

# 3. Draw a circle
# cv.circle(image, center, radius, color, thickness)
cv.circle(blank, (blank.shape[1]//2, blank.shape[0]//2), 100, (0,0,255), thickness=2)
cv.imshow('Circle', blank)

# 4. Draw a line
# cv.line(image, pt1, pt2, color, thickness)
cv.line(blank, (0,0), (blank.shape[1], blank.shape[0]), (255,0,0), thickness=2)
cv.imshow('Line', blank)

# 5. Write text 
# cv.putText(image, string, origin, fontFace, fontScale, color, trhickness)
cv.putText(blank, 'Open CV', (225,225), cv.FONT_HERSHEY_TRIPLEX, 1.0, (250,0,250), thickness=2)
cv.imshow('Text', blank)

cv.waitKey(0)