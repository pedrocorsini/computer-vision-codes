import cv2 as cv
import numpy as np

img = cv.imread('resources/Photos/park.jpg')
cv.imshow('Park', img)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Gray', gray)

# Computing Edges

# Laplacian
lap = cv.Laplacian(gray, ddepth=cv.CV_64F)
lap = np.uint8(np.absolute(lap))
cv.imshow('Laplacian', lap)

# Sobel 
sobelx = cv.Sobel(gray, cv.CV_64F, dx=1, dy=0)
sobely = cv.Sobel(gray, cv.CV_64F, dx=0, dy=1)
combined_sobel = cv.bitwise_or(sobelx, sobely)

cv.imshow('Sobel X', sobelx)
cv.imshow('Sobel Y', sobely)
cv.imshow('Combined Sobel', combined_sobel)

canny = cv.Canny(img, 150, 175)
cv.imshow('Canny', canny)

cv.waitKey(0)