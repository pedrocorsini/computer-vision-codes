import cv2 as cv
# import matplotlib.pyplot as plt

img = cv.imread('resources/Photos/park.jpg')
cv.imshow('Park', img)

# plt.imshow(img)
# plt.show()

# Color Spaces
# BGR to GrayScale

gray = cv.cvtColor(src=img, code=cv.COLOR_BGR2GRAY)
cv.imshow('Gray', gray)

# Convert from BGR to HSV (Hue Saturation Value)
hsv = cv.cvtColor(src=img, code=cv.COLOR_BGR2HSV)
cv.imshow('HSV', hsv)

# BGR to LAB (L*a*b)
lab = cv.cvtColor(img, cv.COLOR_BGR2LAB)
cv.imshow('LAB', lab)

# BGR to RGB
rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
cv.imshow('RGB', rgb)

# plt.imshow(rgb)
# plt.title('RGB')
# plt.show()

# HSV to BGR
hsv_bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
cv.imshow('HSV to BGR', hsv_bgr)

# LAB to BGR
lab_bgr = cv.cvtColor(lab, cv.COLOR_LAB2BGR)
cv.imshow('LAB to BGR', lab_bgr)

cv.waitKey(0)