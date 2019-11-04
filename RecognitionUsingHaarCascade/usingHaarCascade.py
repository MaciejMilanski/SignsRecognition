import cv2 as cv
import numpy as np
from PIL import Image
import pytesseract


def printRectangle(sign, image, index, offsetx, offsety): # offsets are needed to change start point of coordinate system, becouse we want ot print rectangle on full size image, but we hav coordinate from croped image
    for (x, y, w, h) in sign:
        cv.rectangle(image, (x + offsetx, y + offsety), (x + offsetx + w, y + offsety + h), (0, 0, 255), 10)
        flag = 1
        return flag

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
    mask = cv.resize(mask,(200,200))
    mask = mask[50: 150, 50: 150]
    mask = cv.resize(mask,(50,50))
    #mask = cv.bitwise_not(mask)
    cv.imshow('mask', mask)
    return mask

def putOnTable(image):
    table = cv.imread("table.jpg",0)
    xOffset = 10
    yOffset = 10
    table[yOffset:yOffset + image.shape[0], xOffset:xOffset + image.shape[1]] = image
    table = cv.bitwise_not(table)
    cv.imshow('table',table)
    cv.imwrite("text.jpg",table)

def readText():
    textImg = Image.open("text.jpg")
    text = pytesseract.image_to_string(textImg, lang='eng')
    return text

def checkIftextIsNumber(text):
    try:
        int(text)
    except ValueError:
        return False

def checkIfTextPossibyIsSpeedLimit(text):
    if text % 5 == 0:
        return True
    else:
        return False

cap = cv.VideoCapture(0)

#speed limits
speed_limit_10_cascade = cv.CascadeClassifier('speed_limit_cascades/speed_limit_10_cascade.xml')

#coordinates or crop fcn
cropX = 500
cropY = 100

while(1):


    # Take each frame
    _, BGRimg1 = cap.read()
    #BGRimg1 = cv.imread('image.jpg', 1)
    #cv.imshow('RecognitionRGB', BGRimg1)
    BGRimg1 = cv.resize(BGRimg1,(800,640))

    # Convert BGR to GRAY
    GRAYimg1 = cv.cvtColor(BGRimg1,cv.COLOR_BGR2GRAY)
    #crop image
    cropGRAYimg1 = GRAYimg1[cropY:cropY + 300 , cropX:cropX + 300]
    speed_limit_10 = speed_limit_10_cascade.detectMultiScale(cropGRAYimg1, 1.3, 5)

    flag = printRectangle(speed_limit_10, BGRimg1, 10, cropX, cropY)
    if flag == 1:
        cropSignImg = cropSign(speed_limit_10, BGRimg1, cropX, cropY)
        mask = filter(cropSignImg)
        putOnTable(mask)
        text = readText()
        numberFlag = checkIftextIsNumber(text)
        #speedLimitFlag = checkIfTextPossibyIsSpeedLimit(int(text))
        #if numberFlag == True:# and speedLimitFlag == True:
        print(text)

    cv.imshow('cropGRAYimg1', cropGRAYimg1)
    cv.imshow('RecognitionRGB', BGRimg1)
    k = cv.waitKey(5) & 0xFF
    if k == 27:
        break
    #cv.waitKey(9999)
cv.destroyAllWindows()