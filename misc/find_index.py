import cv2
import sys

idx = sys.argv[1]
cap = cv2.VideoCapture(int(idx))

while True:
    ret, img = cap.read()

    cv2.imshow('idx_finder', img)
    if cv2.waitKey(100) == ord('q'):
        break