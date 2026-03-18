import cv2 as cv

img = cv.imread('resources/Photos/cats.jpg')
cv.imshow('Cats', img)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Gray', gray)

# Simple Thresholding
threshold, thresh = cv.threshold(gray, thresh=150, maxval=255, type=cv.THRESH_BINARY)
cv.imshow('Simple Thresholded', thresh)
# print(threshold)

threshold, thresh_inv = cv.threshold(gray, thresh=150, maxval=255, type=cv.THRESH_BINARY_INV)
cv.imshow('Simple Thresholded Inverse', thresh_inv)
# print(threshold)

# Adaptive Tresholding
adaptive_thresh = cv.adaptiveThreshold(gray, 
                                       maxValue=255, 
                                       adaptiveMethod=cv.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                       thresholdType=cv.THRESH_BINARY_INV, 
                                       blockSize=11, 
                                       C=9)
cv.imshow('Adaptive Threshold', adaptive_thresh)

cv.waitKey(0)