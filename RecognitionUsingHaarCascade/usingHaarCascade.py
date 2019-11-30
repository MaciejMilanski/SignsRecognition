import cv2 as cv
import numpy as np


def printRectangle(sign, image, index, offsetx, offsety): # offsets are needed to change start point of coordinate system, becouse we want ot print rectangle on full size image, but we hav coordinate from croped image
    for (x, y, w, h) in sign:
        cv.rectangle(image, (x + offsetx, y + offsety), (x + offsetx + w, y + offsety + h), (0, 0, 255), 10)
        roi_color = image[y:y + h, x:x + w]
        print(index)
def cropSign(sign, image, offsetx, offsety):
    for (x, y, w, h) in sign:
        croppedSign = image[y + offsety : y + offsety + h, x + offsetx : x + offsetx + w]
        cv.imshow('cropped text', croppedSign)
        return croppedSign

def putOnTable(image):
    table = cv.imread("table.jpg",0)
    xOffset = 100
    yOffset = 100
    table[yOffset:yOffset + image.shape[0], xOffset:xOffset + image.shape[1]] = image
    #table = cv.bitwise_not(table)
    cv.imshow('table',table)
    cv.imwrite("text.jpg",table)
    return table

def filter(image):
    HSVimage = cv.cvtColor(image,cv.COLOR_BGR2HSV)
    lowBlack = np.array([200, 150, 0])
    highBlack = np.array([255, 255, 97])
    mask = cv.inRange(HSVimage, lowBlack, highBlack)
    #mask = cv.resize(mask,(200,200))
    #mask = mask[50: 150, 50: 150]
    #mask = cv.resize(mask,(50,50))
    #mask = cv.bitwise_not(mask)
    cv.imshow('mask', mask)
    return mask

cap = cv.VideoCapture('MOVI4128.avi')
#cap = cv.VideoCapture(0)

#speed limits
speed_limit_5_cascade = cv.CascadeClassifier('speed_limit_cascades/speed_limit_5_cascade.xml')
speed_limit_10_cascade = cv.CascadeClassifier('speed_limit_cascades/speed_limit_10_cascade.xml')
speed_limit_20_cascade = cv.CascadeClassifier('speed_limit_cascades/speed_limit_20_cascade.xml')
speed_limit_30_cascade = cv.CascadeClassifier('speed_limit_cascades/speed_limit_30_cascade.xml')
speed_limit_40_cascade = cv.CascadeClassifier('speed_limit_cascades/speed_limit_40_cascade.xml')
speed_limit_50_cascade = cv.CascadeClassifier('speed_limit_cascades/speed_limit_50_cascade.xml')
speed_limit_60_cascade = cv.CascadeClassifier('speed_limit_cascades/speed_limit_60_cascade.xml')
#coordinates or crop fcn
cropX = 250
cropY = 250
#min and max Size of detected objects
minSizeXY = 30
maxSizeXY = 70
while(1):


    # Take each frame
    _, BGRimg1 = cap.read()
    #BGRimg1 = cv.imread('image5.jpg',1)
    BGRimg1 = cv.resize(BGRimg1,(800,640))
    # Convert BGR to GRAY
    GRAYimg1 = cv.cvtColor(BGRimg1,cv.COLOR_BGR2GRAY)
    #crop image
    #ret,THRESHimg1 = cv.threshold(GRAYimg1,125,130,cv.THRESH_BINARY)
    #cv.imshow('fdfdf',THRESHimg1)
    cropGRAYimg1 = GRAYimg1[cropY:cropY + 150 , cropX:cropX + 400]
    #cropGRAYimg1 = cv.resize(cropGRAYimg1,(,1000))

    #speed limits
    speed_limit_5 = speed_limit_5_cascade.detectMultiScale(cropGRAYimg1, 1.1, 15, 0, (minSizeXY, minSizeXY),(maxSizeXY,maxSizeXY))
    speed_limit_10 = speed_limit_10_cascade.detectMultiScale(cropGRAYimg1, 1.1, 15, 0, (minSizeXY, minSizeXY),(maxSizeXY,maxSizeXY))
    speed_limit_20 = speed_limit_20_cascade.detectMultiScale(cropGRAYimg1, 1.1, 15, 0, (minSizeXY, minSizeXY),(maxSizeXY,maxSizeXY))
    speed_limit_30 = speed_limit_30_cascade.detectMultiScale(cropGRAYimg1, 1.1, 15, 0, (minSizeXY, minSizeXY),(maxSizeXY,maxSizeXY))
    speed_limit_40 = speed_limit_40_cascade.detectMultiScale(cropGRAYimg1, 1.1, 15, 0, (minSizeXY, minSizeXY),(maxSizeXY,maxSizeXY))
    speed_limit_50 = speed_limit_50_cascade.detectMultiScale(cropGRAYimg1, 1.20, 15, 0, (minSizeXY, minSizeXY),(maxSizeXY,maxSizeXY))
    speed_limit_60 = speed_limit_60_cascade.detectMultiScale(cropGRAYimg1, 1.2, 11, 0, (minSizeXY, minSizeXY),(maxSizeXY, maxSizeXY))

    #crop_sign = cropSign(speed_limit_10, GRAYimg1, cropX, cropY)
    # text = putOnTable(crop_sign)
    # speed_limit_30 = speed_limit_30_cascade.detectMultiScale(text, 1.3, 8)
    #filter(cropGRAYimg1)

    #speed limits
    #printRectangle(speed_limit_5, BGRimg1, 5, cropX, cropY)
    #printRectangle(speed_limit_10, BGRimg1, 10, cropX, cropY)
    #printRectangle(speed_limit_20, BGRimg1, 20, cropX, cropY)
    #printRectangle(speed_limit_30, BGRimg1, 30, cropX, cropY)
    #printRectangle(speed_limit_40, BGRimg1, 40, cropX, cropY)
    printRectangle(speed_limit_50, BGRimg1, 50, cropX, cropY)
    printRectangle(speed_limit_60, BGRimg1, 60, cropX, cropY)

    cv.imshow('cropGRAYimg1', cropGRAYimg1)
    cv.imshow('RecognitionRGB', BGRimg1)
    k = cv.waitKey(5) & 0xFF
    if k == 27:
        break
    #cv.waitKey(1000)
cv.destroyAllWindows()
