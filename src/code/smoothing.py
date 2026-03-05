import cv2 as cv

img = cv.imread('resources/Photos/cats.jpg')
cv.imshow('Cats', img)

# Blurring

# Averaging 
average = cv.blur(src=img, ksize=(3,3))
cv.imshow('Average Blur', average)

# Gaussian Blur
gaussian = cv.GaussianBlur(img, (3,3), sigmaX=0)
cv.imshow('Gaussian Blur', gaussian)

# Median Blur
median = cv.medianBlur(img, 3)
cv.imshow('Median Blur', median)

# Bilateral 
bilateral = cv.bilateralFilter(img, 10, 35, 25)
cv.imshow('Bilateral Filtering', bilateral)

cv.waitKey(0)