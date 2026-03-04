import cv2 as cv
import numpy as np



img = cv.imread('resources/Photos/park.jpg')
cv.imshow('Park', img)

blank = np.zeros(img.shape[:2], dtype='uint8')  

# Split color channels
b, g, r = cv.split(img)

# cv.imshow('Blue', b)
# cv.imshow('Green', g)
# cv.imshow('Red', r)

print(img.shape)
print(b.shape)
print(g.shape)
print(r.shape)

# Merge color channels
merged = cv.merge([b, g, r])
cv.imshow('Merged', merged)

blue = cv.merge([b, blank, blank])
green = cv.merge([blank, g, blank])
red = cv.merge([blank, blank, r])

cv.imshow('Blue Merged', blue)
cv.imshow('Green Merged', green)
cv.imshow('Red Merged', red)

cv.waitKey(0)