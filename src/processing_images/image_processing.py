import cv2 as cv
import numpy as np

class ImageProcessing:
    def __init__(self, image_path):
        self.image_path = image_path
        self._image = cv.imread(self.image_path)

    def read(self):
        self._image = cv.imread(self.image_path)

        if self.image_path is None:
            raise FileNotFoundError(f'Image not found in {self.image_path}') 
        return self

    def grayscale(self):
        if self._image is not None:
            self._image = cv.cvtColor(self._image, cv.COLOR_BGR2GRAY)
        return self

    def simple_gaussian(self, ksize):
        if len(ksize) != 2:
            raise ValueError('ksize must be iterable with 2 elements (x,y)')
        ksize_tuple = (int(ksize[0]), int(ksize[1]))

        if self._image is not None:
            self._image = cv.GaussianBlur(self._image, ksize_tuple, sigmaX=0)
        return self
    
    def display(self, name):

        if name is None:
            raise ValueError('name empty')

        if not isinstance(name, str):
            raise TypeError(f'name ({name}) must be a string')

        if self._image is not None:
            cv.imshow(name, self._image)
        return self
    
    def rectangle(self, pt1, pt2, color, thickness):
        if len(pt1) != 2 and pt2 != 2:
            raise ValueError(f'pt1 and pt2 must be iterable with 2 elements (x,y)')
        p1 = (int(pt1[0]), int(pt1[1]))
        p2 = (int(pt2[0]), int(pt2[1]))
        
        if len(color) != 3:
            raise ValueError(f'color must be iterable with 3 elements (b, g, r)')
        
        color_tuple = (int(color[0]), int(color[1]), int(color[2]))

        if thickness is None:
            thickness = -1

        if self._image is not None:
            self._image = cv.rectangle(self._image, p1, p2, color_tuple, thickness=thickness)
        return self
    
    def threshold_binary(self, thresh_value, max_value):
        if self._image is not None:
            self._image = cv.cvtColor(self._image, cv.COLOR_BGR2GRAY)
            threshold, self._image = cv.threshold(self._image, thresh=thresh_value, maxval=max_value, type=cv.THRESH_BINARY)
        return self
    
    def close(self):
        cv.waitKey(0)
        cv.destroyAllWindows()
