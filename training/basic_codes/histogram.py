import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

# pip install matplotlib

img = cv.imread('resources/Photos/cats.jpg')
cv.imshow('Cats', img)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# cv.imshow('Gray', gray)

blank = np.zeros(img.shape[:2], dtype='uint8')

mask = cv.circle(blank.copy(), (img.shape[1]//2, img.shape[0]//2), 100, 255, -1)
masked = cv.bitwise_and(img, img, mask=mask)
cv.imshow('Mask', masked)

# # Grayscale histogram

# gray_hist = cv.calcHist([gray], [0], mask, [256], [0, 256])

# plt.figure()
# plt.title('Grayscale Histogram')
# plt.xlabel('Bins')
# plt.ylabel('# of pixles')
# plt.plot(gray_hist)
# plt.xlim([0,256])
# plt.show()

# Color Histogram

colors = ('b', 'g', 'r')
plt.figure()
plt.title('Color Histogram')
plt.xlabel('Bins')
plt.ylabel('# of pixles')
for i,col in enumerate(colors):
    hist = cv.calcHist([img], [i], mask, [256], [0, 256])
    plt.plot(hist, color=col)
    plt.xlim([0, 256])
plt.show()

cv.waitKey(0)