import cv2
import numpy as np


faser = cv2.CascadeClassifier('./haarcascades/haarcascade_frontalface_default.xml')

img = cv2.imread('oic.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = faser.detectMultiScale(gray, 1.1, 4)

for (x, y, w, h) in faces:
    cv2.rectangle(img, (x,y), (x+w, y+h), (255, 0, 0), 2)

cv2.imshow('img', img)
cv2.waitKey()