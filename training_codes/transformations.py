import cv2 as cv
import numpy as np

img = cv.imread('resources/Photos/park.jpg')
cv.imshow('Boston', img)

# Translation - Shifiting the image among the X and Y axes
def translate(img, x, y): # x - y -> numbers of pixels you want to shift 
    transMat = np.float32([[1,0,x],[0,1,y]])
    dimensions = (img.shape[1], img.shape[0])
    return cv.warpAffine(img, transMat, dimensions)

# -x --> left
# -y --> up
# x --> right
# y --> down

translated = translate(img, -100, 100)
cv.imshow('Translated', translated)

# Rotation

def rotate(img, angle, rotPoint=None):
    (height, width) = img.shape[:2]

    if rotPoint is None:
        rotPoint = (width//2, height//2)
    
    rotMat = cv.getRotationMatrix2D(rotPoint, angle, 1.0)
    dimensions = (width, height)

    return cv.warpAffine(img, rotMat, dimensions)

rotated = rotate(img, -45)
cv.imshow('Rotated', rotated)

rotated_90 = rotate(img, -90)
cv.imshow('Rotated 90', rotated_90)

# Resizing

# shrink = (cv.INTER_AREA) | enlarge = (cv.INTER_LINEAR) or (cv.INTER_CUBIC) 
resized = cv.resize(img, (500,500), interpolation=cv.INTER_CUBIC) 
cv.imshow('Resized', resized)

# Flipping

# 0 - vertically
# 1 - horizontally
# -1 - both vertical and horizontal
flip = cv.flip(img, -1)
cv.imshow('Flip', flip)

# Cropping

cropped = img[200:400, 300:400]
cv.imshow('Cropped', cropped)

cv.waitKey(0)
