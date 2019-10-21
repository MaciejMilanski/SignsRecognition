import cv2 as cv
import numpy as np


def printRectangle(speed_limit, image, index):
    for (x, y, w, h) in speed_limit:
        cv.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 10)
        roi_color = image[y:y + h, x:x + w]
        print(index)




cap = cv.VideoCapture(0)

speed_limit_5_cascade = cv.CascadeClassifier('speed_limit_cascades/speed_limit_5_cascade.xml')
speed_limit_10_cascade = cv.CascadeClassifier('speed_limit_cascades/speed_limit_10_cascade.xml')
speed_limit_20_cascade = cv.CascadeClassifier('speed_limit_cascades/speed_limit_20_cascade.xml')
speed_limit_30_cascade = cv.CascadeClassifier('speed_limit_cascades/speed_limit_30_cascade.xml')
speed_limit_40_cascade = cv.CascadeClassifier('speed_limit_cascades/speed_limit_40_cascade.xml')

while(1):


    # Take each frame
    _, BGRimg1 = cap.read()
    BGRimg1 = cv.resize(BGRimg1,(800,640))
    # Convert BGR to GRAY
    GRAYimg1 = cv.cvtColor(BGRimg1,cv.COLOR_BGR2GRAY)
    #crop image
    cropGRAYimg1 = GRAYimg1[100:100 + 300 , 500:500 + 300]


    speed_limit_5 = speed_limit_5_cascade.detectMultiScale(cropGRAYimg1, 1.3, 5)
    speed_limit_10 = speed_limit_10_cascade.detectMultiScale(cropGRAYimg1, 1.3, 5)
    speed_limit_20 = speed_limit_20_cascade.detectMultiScale(cropGRAYimg1, 1.3, 5)
    speed_limit_30 = speed_limit_30_cascade.detectMultiScale(cropGRAYimg1, 1.3, 5)
    speed_limit_40 = speed_limit_40_cascade.detectMultiScale(cropGRAYimg1, 1.3, 5)

    printRectangle(speed_limit_5, BGRimg1, 5)
    printRectangle(speed_limit_10, BGRimg1, 10)
    printRectangle(speed_limit_20, BGRimg1, 20)
    printRectangle(speed_limit_30, BGRimg1, 30)
    printRectangle(speed_limit_40, BGRimg1, 40)

    cv.imshow('cropGRAYimg1', cropGRAYimg1)
    cv.imshow('RecognitionRGB', BGRimg1)
    k = cv.waitKey(5) & 0xFF
    if k == 27:
        break

cv.destroyAllWindows()
