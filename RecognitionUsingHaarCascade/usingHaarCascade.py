import cv2 as cv
import numpy as np

def printRectangle(speed_limit, image, index):
    for (x, y, w, h) in speed_limit:
        cv.rectangle(RGBimg1, (x, y), (x + w, y + h), (0, 0, 255), 10)
        roi_color = RGBimg1[y:y + h, x:x + w]
        print(index)




cap = cv.VideoCapture(0)
speed_limit_5_cascade = cv.CascadeClassifier('speed_limit_cascades/speed_limit_5_cascade.xml')
speed_limit_10_cascade = cv.CascadeClassifier('speed_limit_cascades/speed_limit_10_cascade.xml')
speed_limit_20_cascade = cv.CascadeClassifier('speed_limit_cascades/speed_limit_20_cascade.xml')
speed_limit_30_cascade = cv.CascadeClassifier('speed_limit_cascades/speed_limit_30_cascade.xml')
speed_limit_40_cascade = cv.CascadeClassifier('speed_limit_cascades/speed_limit_40_cascade.xml')
speed_limit_50_cascade = cv.CascadeClassifier('speed_limit_cascades/speed_limit_50_cascade.xml')
while(1):
    _, RGBimg1 = cap.read()
    #cv.imshow('A7RecognitionRGB', RGBimg1)
    # Take each frame


    # Convert BGR to GRAY
    GRAYimg1 = cv.cvtColor(RGBimg1,cv.COLOR_BGR2GRAY)

    speed_limit_5 = speed_limit_5_cascade.detectMultiScale(GRAYimg1, 1.3, 5)
    speed_limit_10 = speed_limit_10_cascade.detectMultiScale(GRAYimg1, 1.3, 5)
    speed_limit_20 = speed_limit_20_cascade.detectMultiScale(GRAYimg1, 1.3, 5)
    speed_limit_30 = speed_limit_30_cascade.detectMultiScale(GRAYimg1, 1.3, 5)
    speed_limit_40 = speed_limit_40_cascade.detectMultiScale(GRAYimg1, 1.3, 5)
    speed_limit_50 = speed_limit_50_cascade.detectMultiScale(GRAYimg1, 1.3, 5)

    printRectangle(speed_limit_5, RGBimg1, 5)
    printRectangle(speed_limit_10, RGBimg1, 10)
    printRectangle(speed_limit_20, RGBimg1, 20)
    printRectangle(speed_limit_30, RGBimg1, 30)
    printRectangle(speed_limit_40, RGBimg1, 40)
    printRectangle(speed_limit_50, RGBimg1, 50)

    cv.imshow('A7RecognitionRGB', RGBimg1)
    k = cv.waitKey(5) & 0xFF
    if k == 27:
        break

cv.destroyAllWindows()
