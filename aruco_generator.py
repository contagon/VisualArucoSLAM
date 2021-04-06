import cv2
import argparse

def save_markers(dictionary, num, folder):
    for i in range(num):
        img = cv2.aruco.drawMarker(dictionary, i, 250)
        cv2.imwrite(f'{folder}/marker{i}.jpg', img)

# parse args
ap = argparse.ArgumentParser()
ap.add_argument("-n", "--num", type=int, required=True, help="Num of tags to save")
ap.add_argument("-f", "--folder", type=str, default="markers", help="Folder to save them in")
args = vars(ap.parse_args())

# choose dictionary from https://docs.opencv.org/master/d9/d6a/group__aruco.html#gac84398a9ed9dd01306592dd616c2c975
dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_APRILTAG_36h11)

save_markers(dictionary, **args)