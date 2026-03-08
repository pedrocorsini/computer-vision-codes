import os
import cv2 as cv
import numpy as np

people = ['Ben Afflek', 'Elton John', 'Jerry Seinfield', 'Madonna', 'Mindy Kaling']
DIR = r'D:\repositories\opencv-learning\resources\Faces\train' # Change the path if needed

haar_cascade = cv.CascadeClassifier('src/face_detection/haar_cascade_detection/haar_face.xml')

features = []
labels = []

def create_train():
    for person in people:
        path = os.path.join(DIR, person)
        label = people.index(person)

        for img in os.listdir(path):
            img_path = os.path.join(path, img)

            img_array = cv.imread(img_path)
            gray = cv.cvtColor(img_array, cv.COLOR_BGR2GRAY)

            faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

            for (x,y,w,h) in faces_rect:
                faces_roi = gray[y:y+h, x:x+w]
                features.append(faces_roi)
                labels.append(label)

create_train()

# print(f'Length of the features list = {len(features)}')
# print(f'Length of the labels = {len(labels)}')

print('-------------- Training done --------------')

features = np.array(features, dtype='object')
labels = np.array(labels)

faces_recognizer = cv.face.LBPHFaceRecognizer_create()

# Train the Recognizer on the features list and labels list

faces_recognizer.train(features, labels)

faces_recognizer.save('src/face_detection/face_recognition/face_trained.yml')
np.save('src/face_detection/face_recognition/features.npy', features)
np.save('src/face_detection/face_recognition/labels.npy', labels)