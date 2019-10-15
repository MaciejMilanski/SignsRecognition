import cv2 as cv
import numpy as np

cap = cv.VideoCapture(0)
speed_limit_50_cascade = cv.CascadeClassifier('speed_limit_50_stage9_cascade.xml')
while(1):
    _, RGBimg1 = cap.read()
    #cv.imshow('A7RecognitionRGB', RGBimg1)
    # Take each frame


    # Convert BGR to GRAY
    GRAYimg1 = cv.cvtColor(RGBimg1,cv.COLOR_BGR2GRAY)
    speed_limit_50 = speed_limit_50_cascade.detectMultiScale(GRAYimg1, 1.3, 5)

    for (x, y, w, h) in speed_limit_50:
        cv.rectangle(RGBimg1, (x, y), (x + w, y + h), (0, 0, 255), 10)
        roi_color = RGBimg1[y:y + h, x:x + w]
    cv.imshow('A7RecognitionRGB', RGBimg1)
    k = cv.waitKey(5) & 0xFF
    if k == 27:
        break

cv.destroyAllWindows()
