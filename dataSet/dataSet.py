import cv2 as cv
import numpy as np
import os


def rawImages():
    #list = os.listdir('raw_images')

    if not os.path.exists('negatives'):
        print("niema")

    picNum = 1

    for filename in os.listdir('negatives'):
        try:
            print(picNum)
            filename = 'negatives/' + filename
            dst = 'negatives/' + str(picNum) + ".jpg"
            src = filename
            print(src)
            print(dst)

            os.rename(src, dst)

            img = cv.imread("negatives/" + str(picNum) + '.jpg', 0)
            resizedImage = cv.resize(img, (100, 100))
            cv.imwrite("negatives/" + str(picNum) + '.jpg', resizedImage)
            picNum += 1

        except Exception as e:
            print(str(e))
def createBgFile():
    for file_type in ['negatives']:
        for img in os.listdir(file_type):
            if file_type == 'negatives':
                line = file_type + '/' + img + '\n'
                with open('bg.txt','a') as f:
                    f.write(line)
def samplePrep():
    sample = cv.imread("sample/sample.jpg",0)
    resizedSample = cv.resize(sample,(50,50))
    cv.imwrite("sample/sample5050.jpg", resizedSample)
#rawImages()
createBgFile()
#samplePrep()