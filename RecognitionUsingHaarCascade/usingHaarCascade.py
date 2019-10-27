import cv2 as cv
import numpy as np
from PIL import ImageOps


def printRectangle(sign, image, index, offsetx, offsety): # offsets are needed to change start point of coordinate system, becouse we want ot print rectangle on full size image, but we hav coordinate from croped image
    for (x, y, w, h) in sign:
        cv.rectangle(image, (x + offsetx, y + offsety), (x + offsetx + w, y + offsety + h), (0, 0, 255), 10)
        print(index)

def cropSign(sign, image, offsetx, offsety):
    for (x, y, w, h) in sign:
        croppedSign = image[y + offsety : y + offsety + h, x + offsetx : x + offsetx + w]
        cv.imshow('cropped text', croppedSign)
        return croppedSign

def filter(image):
    HSVimage = cv.cvtColor(image,cv.COLOR_BGR2HSV)
    lowBlack = np.array([0, 0, 0])
    highBlack = np.array([255, 255, 120])
    mask = cv.inRange(HSVimage, lowBlack, highBlack)
    #mask = ImageOps.invert(mask)
    cv.imshow('mask', mask)
    return mask

cap = cv.VideoCapture(0)
#warning signs
A7_cascade = cv.CascadeClassifier('warning_signs_cascades/A7_cascade.xml')
#information signs
D2_cascade = cv.CascadeClassifier('information_signs_cascades/D2_cascade.xml')
#speed limits
speed_limit_5_cascade = cv.CascadeClassifier('speed_limit_cascades/speed_limit_5_cascade.xml')
speed_limit_10_cascade = cv.CascadeClassifier('speed_limit_cascades/speed_limit_10_cascade.xml')
speed_limit_20_cascade = cv.CascadeClassifier('speed_limit_cascades/speed_limit_20_cascade.xml')
speed_limit_30_cascade = cv.CascadeClassifier('speed_limit_cascades/speed_limit_30_cascade.xml')
speed_limit_40_cascade = cv.CascadeClassifier('speed_limit_cascades/speed_limit_40_cascade.xml')

#coordinates or crop fcn
cropX = 500
cropY = 100

while(1):


    # Take each frame
    #_, BGRimg1 = cap.read()
    BGRimg1 = cv.imread('image.jpg', 1)
    cv.imshow('RecognitionRGB', BGRimg1)
    BGRimg1 = cv.resize(BGRimg1,(800,640))

    # Convert BGR to GRAY
    GRAYimg1 = cv.cvtColor(BGRimg1,cv.COLOR_BGR2GRAY)
    #crop image
    cropGRAYimg1 = GRAYimg1[cropY:cropY + 300 , cropX:cropX + 300]

    #warning_signs
    A7 = A7_cascade.detectMultiScale(cropGRAYimg1, 1.3, 5)

    #information signs
    D2 = D2_cascade.detectMultiScale(cropGRAYimg1, 1.3, 5)

    #speed limits
    speed_limit_5 = speed_limit_5_cascade.detectMultiScale(cropGRAYimg1, 1.3, 5)
    speed_limit_10 = speed_limit_10_cascade.detectMultiScale(cropGRAYimg1, 1.3, 5)
    speed_limit_20 = speed_limit_20_cascade.detectMultiScale(cropGRAYimg1, 1.3, 5)
    speed_limit_30 = speed_limit_30_cascade.detectMultiScale(cropGRAYimg1, 1.3, 5)
    speed_limit_40 = speed_limit_40_cascade.detectMultiScale(cropGRAYimg1, 1.3, 5)

    cropSignImg = cropSign(speed_limit_10, BGRimg1, cropX, cropY)
    filter(cropSignImg)
    #warning signs
    printRectangle(A7, BGRimg1, "A7", cropX, cropY)

    #information signs
    printRectangle(D2, BGRimg1, "D2", cropX, cropY)

    #speed limits
    printRectangle(speed_limit_5, BGRimg1, 5, cropX, cropY)
    printRectangle(speed_limit_10, BGRimg1, 10, cropX, cropY)
    printRectangle(speed_limit_20, BGRimg1, 20, cropX, cropY)
    printRectangle(speed_limit_30, BGRimg1, 30, cropX, cropY)
    printRectangle(speed_limit_40, BGRimg1, 40, cropX, cropY)

    cv.imshow('cropGRAYimg1', cropGRAYimg1)
    cv.imshow('RecognitionRGB', BGRimg1)
    k = cv.waitKey(5) & 0xFF
    if k == 27:
        break
    cv.waitKey(9999)
cv.destroyAllWindows()
