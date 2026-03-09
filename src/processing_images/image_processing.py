import cv2 as cv
import numpy as np

# Processing Pictures Class

class PictureProcessing:
    def __init__(self, image_path):
        self.image_path = image_path
        self._image = None        

    def read_picture(self):
        self._image = cv.imread(self.image_path)

        if self._image is None:
            raise FileNotFoundError(f'Picture not found in {self.image_path}') 
        
        return self

    def grayscale(self):
        if self._image is not None:
            self._image = cv.cvtColor(self._image, cv.COLOR_BGR2GRAY)

        return self

    def gaussian_blur(self, ksize):
        if len(ksize) != 2:
            raise ValueError('ksize must be iterable with 2 elements (x,y)')
        ksize_tuple = (int(ksize[0]), int(ksize[1]))

        if self._image is not None:
            self._image = cv.GaussianBlur(self._image, ksize_tuple, sigmaX=0)

        return self
    
    def display(self, name):
        if not isinstance(name, str):
            raise TypeError(f'name ({name}) must be a string')

        if self._image is not None:
            cv.imshow(name, self._image)

        return self
    
    def rectangle(self, pt1, pt2, color, thickness):

        if len(pt1) != 2 or len(pt2) != 2:
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
            if len(self._image.shape) == 3:
                self._image = cv.cvtColor(self._image, cv.COLOR_BGR2GRAY)
            self._image = cv.threshold(self._image, thresh=thresh_value, maxval=max_value, type=cv.THRESH_BINARY)[1]

        return self
    
    def canny(self, thresh1, thresh2):
        if self._image is not None:
            self._image = cv.Canny(self._image, thresh1, thresh2)
        return self

    def close_image(self):
        cv.waitKey(0)
        cv.destroyAllWindows()
        return self

class VideoProcessing:
    def __init__(self, video_path):
        self.video_path = video_path
        self._video = None
        self.operations = []
        
    def read_video(self):
        self._video = cv.VideoCapture(self.video_path)
        
        if not self._video.isOpened():
            raise FileNotFoundError(f'Video not found in {self.video_path}')
        
        return self
       
    def grayscale(self):
        self.operations.append(lambda frame: cv.cvtColor(frame, cv.COLOR_BGR2GRAY))
        return self
    
    def gaussian_blur(self, ksize):
        if len(ksize) != 2:
            raise ValueError('ksize must be iterable with 2 elements (x,y)')
        ksize_tuple = (int(ksize[0]), int(ksize[1]))    
        self.operations.append(lambda frame: cv.GaussianBlur(frame, ksize_tuple, 0))   

        return self 
    
    def flip_video(self, option):
        # 0 - vertically
        # 1 - horizontally
        # -1 - both vertical and horizontal
        self.operations.append(lambda frame: cv.flip(frame, option))
        return self
    
    def thresh(self, thresh_value, max_value):
        self.operations.append(lambda frame: cv.threshold(frame, thresh_value, max_value, type=cv.THRESH_BINARY)[1])

        return self
    
    def canny(self, thresh1, thresh2):
        self.operations.append(lambda frame: cv.Canny(frame, thresh1, thresh2))

        return self


    def display(self, name):

        if not isinstance(name, str):
            raise TypeError(f'name ({name}) must be a string')

        if self._video is None:
            raise ValueError('Video not loaded. Call read_video() first.')

        while True:
            isTrue, frame = self._video.read()

            if not isTrue or frame is None:
                break

            for op in self.operations:
                frame = op(frame)

            cv.imshow(name, frame)

            if cv.waitKey(20) & 0xFF == ord('d'): #if the user press 'd' on the keyboard, breaks the loop
                break

        self._video.release()
        cv.destroyAllWindows()
        return self
