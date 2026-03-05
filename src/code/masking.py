import cv2 as cv
import numpy as np

img = cv.imread('resources/Photos/cats 2.jpg')
cv.imshow('Cats 2', img)

# Creating the blank img for masking -> It needs to be the same size of the original img

blank = np.zeros(img.shape[:2], dtype='uint8')
# cv.imshow('Blank', blank)

mask = cv.circle(blank.copy(), (img.shape[1]//2, img.shape[0]//2), 100, 255, -1)
# cv.imshow('Mask', mask)

mask_rec = cv.rectangle(blank.copy(), (img.shape[1]//2 - 100, img.shape[0]//2 - 20), (img.shape[1]//2 + 20, img.shape[0]//2 + 100), 255, -1)

masked_circle = cv.bitwise_and(img, img, mask=mask)
cv.imshow('Masked Circle', masked_circle)

masked_rectangle = cv.bitwise_and(img, img, mask=mask_rec)
cv.imshow('Masked Rectangle', masked_rectangle)

weird_mask = cv.bitwise_and(mask, mask_rec)
# cv.imshow('Weird Mask', weird_mask)

masked_weird = cv.bitwise_and(img, img, mask=weird_mask)
cv.imshow('Weird Masked Image', masked_weird)

cv.waitKey(0)