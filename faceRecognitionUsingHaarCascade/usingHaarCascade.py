import cv2 as cv
import numpy as np

cap = cv.VideoCapture(0)

while(1):

    # Take each frame
    _, RGBimg1 = cap.read()


    # Convert BGR to GRAY
    GRAYimg1 = cv.cvtColor(RGBimg1,cv.COLOR_BGR2GRAY)

    _, THRESHimg1 = cv.threshold(GRAYimg1, 85, 255, cv.THRESH_BINARY)

    A7 = cv.CascadeClassifier('A7stage5.xml')
    A7Front = A7.detectMultiScale(GRAYimg1, 1.3, 5)




    for (x, y, w, h) in A7Front:
        cv.rectangle(RGBimg1, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = THRESHimg1[y:y + h, x:x + w]
        roi_color = RGBimg1[y:y + h, x:x + w]
    for (x, y, w, h) in A7Front:
        cv.rectangle(RGBimg1, (x, y), (x + w, y + h), (0, 0, 255), 10)
        roi_gray = THRESHimg1[y:y + h, x:x + w]
        roi_color = RGBimg1[y:y + h, x:x + w]
    cv.imshow('A7RecognitionRGB', RGBimg1)
    cv.imshow('A7RecognitionGREY', GRAYimg1)
    cv.imshow('A7RecognitionTHRESH', THRESHimg1)

    k = cv.waitKey(5) & 0xFF
    if k == 27:
        break

cv.destroyAllWindows()
