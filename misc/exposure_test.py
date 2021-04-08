import gtsam
from gtsam.symbol_shorthand import L, X
import numpy as np
import cv2
from plotting import *
import subprocess

cap = cv2.VideoCapture(2)
ret_val , cap_for_exposure = cap.read()

subprocess.check_call("v4l2-ctl -d /dev/video2 -c exposure_auto=1", shell=True)

print(cap.set(cv2.CAP_PROP_EXPOSURE, 100))

while True:
    ret, image = cap.read()
    cv2.imshow('image', image)

    print(cap.get(cv2.CAP_PROP_EXPOSURE))

    if cv2.waitKey(100) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()