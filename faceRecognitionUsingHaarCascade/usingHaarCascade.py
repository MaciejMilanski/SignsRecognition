import cv2 as cv
import numpy as np

cap = cv.VideoCapture(0)

while(1):

    # Take each frame
    _, RGBimg1 = cap.read()


    # Convert BGR to GRAY
    GRAYimg1 = cv.cvtColor(RGBimg1,cv.COLOR_BGR2GRAY)

    _, THRESHimg1 = cv.threshold(GRAYimg1, 85, 255, cv.THRESH_BINARY)

    face_cascade_front = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
    faces_front = face_cascade_front.detectMultiScale(THRESHimg1, 1.3, 5)
    face_cascade_profile = cv.CascadeClassifier('haarcascade_profileface.xml')
    faces_profile = face_cascade_profile.detectMultiScale(THRESHimg1, 1.3, 5)



    for (x, y, w, h) in faces_front:
        cv.rectangle(RGBimg1, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = THRESHimg1[y:y + h, x:x + w]
        roi_color = RGBimg1[y:y + h, x:x + w]
    for (x, y, w, h) in faces_profile:
        cv.rectangle(RGBimg1, (x, y), (x + w, y + h), (0, 0, 255), 10)
        roi_gray = THRESHimg1[y:y + h, x:x + w]
        roi_color = RGBimg1[y:y + h, x:x + w]
    cv.imshow('faceRecognition', RGBimg1)

    k = cv.waitKey(5) & 0xFF
    if k == 27:
        break

cv.destroyAllWindows()
# find better threshold for thresholding and find out if it better way at all