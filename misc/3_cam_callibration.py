import cv2
import os
import argparse
from time import time
import numpy as np
np.set_printoptions(suppress=True) 

"""
This file saves images for 2 chess board images
"""

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 300, 1e-7)

def find_params(files, idxs, output=True):
    img_corners = []
    real_corners = []
    for file in files:
        # Read in image
        image = cv2.imread(file)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Find and improve corners
        ret, corners = cv2.findChessboardCorners(image, board_size, None)
        if ret:
            corners = cv2.cornerSubPix(image, corners, (11,11), (-1,-1), criteria)
            # sometimes they're upside down, fix that
            if corners[0,0,0] > corners[-1,0,0]:
                corners = corners[::-1]

            img_corners.append(corners)
            real_corners.append(idxs)
        else:
            print("Couldn't find corners in ", file)

        if args['display']:
            temp = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            cv2.drawChessboardCorners(temp, board_size, corners, ret)
            cv2.imshow('corners', temp)
            cv2.waitKey()

    # Do actual calibration
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(real_corners, img_corners, image.shape[::-1], None, None)

    if output:
        print("Intrinsic Parameters:")
        print(mtx)
        print("\nDistortion Params: ")
        print(dist.T)

        mean_error = 0
        for i in range(len(real_corners)):
            img_corners2, _ = cv2.projectPoints(real_corners[i], rvecs[i], tvecs[i], mtx, dist)
            error = cv2.norm(img_corners[i], img_corners2, cv2.NORM_L2)/len(img_corners2)
            mean_error += error
        mean_error /= len(real_corners)
        print('\nMean Error:')
        print(mean_error)


    return mtx, dist, img_corners

# Displays countdown and returns last number upon finishing
def countdown(wait, STERIO_BOOL, LEFT_BOOL, CENTER_BOOL, i=None):
    prev = time()
    while wait > 0:
        # take image
        ret, image_l = left.read()
        ret, image_r = right.read()
        ret, image_c = center.read()
        
        # image_r = cv2.rotate(image_r, cv2.ROTATE_180)

        # put countdown on it
        # image, text, location, font, font_scale, color, line_thickness
        cv2.putText(image_r, "R", (550, 50), font, 2, (0, 0, 255), 2)
        cv2.putText(image_l, "L", (550, 50), font, 2, (0, 0, 255), 2)
        cv2.putText(image_c, "C", (550, 50), font, 2, (0, 0, 255), 2)
        cv2.putText(image_r, str(round(wait,1)), (0, 100), font, 4, (0, 0, 255), 2)
        if i is not None:
            cv2.putText(image_r, str(i), (0, 175), font, 2, (0, 0, 255), 2)
        if LEFT_BOOL:
            cv2.putText(image_l, "Cal", (300, 50), font, 2, (0, 0, 255), 2)
        elif CENTER_BOOL:
            cv2.putText(image_c, "Cal", (300, 50), font, 2, (0, 0, 255), 2)
        else:
            cv2.putText(image_r, "Cal", (300, 50), font, 2, (0, 0, 255), 2)
        if STERIO_BOOL:
            cv2.putText(image_c, "Stereo", (100, 50), font, 2, (0, 0, 255), 2)
        # display
        cv2.imshow('calibrate', cv2.hconcat([image_r, image_c, image_l]))
        cv2.waitKey(10)

        # tick countdown
        curr = time()
        if curr - prev >= 0.1:
            prev = curr
            wait = wait - 0.1

    ret, image_l = left.read()
    ret, image_r = right.read()
    ret, image_c = center.read() 
    # image_r = cv2.rotate(image_r, cv2.ROTATE_180)
    return image_l, image_r, image_c    

if __name__ == "__main__":
    # Parse through arguments
    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! NEED TO CHANGE CHESS BOARD SIZE !!!!!!!!!!!!!!!!!!!!!!
    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! NEED TO CHANGE CAMERA INDEX !!!!!!!!!!!!!!!!!!!!!!
    parser = argparse.ArgumentParser(description="Camera Calibrater")
    parser.add_argument("-o", "--outfolder", type=str, default="images", help="Location to save images to")
    parser.add_argument("-ol", "--outfile_left", type=str, default="images/left_params.npz", help="Location to save params")
    parser.add_argument("-or", "--outfile_right", type=str, default="images/right_params.npz", help="Location to save params")
    parser.add_argument("-sr", "--stereo_right_folder", type=str, default="images/Stereo_r", help="Folder to Save and load images from")    
    parser.add_argument("-sl", "--stereo_left_folder", type=str, default="images/Stereo_l", help="Folder to Save and load images from")    
    parser.add_argument("-n", "--num", type=int, default=40, help="Number of images to take")
    parser.add_argument("--start_wait", type=float, default=10, help="Seconds to get setup after starting")
    parser.add_argument("--wait", type=float, default=1, help="Seconds to wait between each image")
    parser.add_argument("-w", "--width", type=int, default=9, help="# of corners wide")
    parser.add_argument("-t", "--height", type=int, default=7, help="# of corners height (tall)")
    parser.add_argument("-d", "--distance", type=float, default=3.88, help="Distance between corners")
    parser.add_argument("--display", action="store_true", help="Display found corners")
    parser.add_argument("-l", "--cam_index_l", type=int, default=2, help="Camera index")
    parser.add_argument("-r", "--cam_index_r", type=int, default=0, help="Camera index")
    parser.add_argument("-c", "--cam_index_c", type=int, default=3, help="Camera index")
    parser.add_argument("-lf", "--left_folder", type=str, default="images/Left_Cali", help="Folder to Save and load images from")
    parser.add_argument("-rf", "--right_folder", type=str, default="images/Right_Cali", help="Folder to Save and load images from")
    parser.add_argument("-cf", "--center_folder", type=str, default="images/Center_Cali", help="Folder to Save and load images from")
   
   
    args = vars(parser.parse_args())

    os.makedirs(args['left_folder'], exist_ok=True)
    os.makedirs(args['right_folder'], exist_ok=True)
    os.makedirs(args['stereo_right_folder'], exist_ok=True)
    os.makedirs(args['stereo_left_folder'], exist_ok=True)
    os.makedirs(args['center_folder'], exist_ok=True)
    os.makedirs(args['outfolder'], exist_ok=True)

    # set a few globals
    left   = cv2.VideoCapture(args['cam_index_l'])
    right  = cv2.VideoCapture(args['cam_index_r'])
    center = cv2.VideoCapture(args['cam_index_c']) 
    font = cv2.FONT_HERSHEY_SIMPLEX 

    #proccess Left images
    for i in range(args['num']):
        if i == 0:
            image_l, image_r, image_c = countdown(args['start_wait'], False, True, False, i+1)
        else:
            image_l, image_r, image_c = countdown(args['wait'], False, True, False, i+1)
            
        #same images
        cv2.imwrite(os.path.join(args['left_folder'], f'L_{i:03d}.jpg'), image_l)
        
    #proccess Right images
    for i in range(args['num']):
        if i == 0:
            image_l, image_r, image_c = countdown(args['start_wait'], False, False, False, i+1)
        else:
            image_l, image_r, image_c = countdown(args['wait'], False, False, False, i+1)
            
        #same images
        cv2.imwrite(os.path.join(args['right_folder'], f'R_{i:03d}.jpg'), image_r)

    #proccess Right images
    for i in range(args['num']):
        if i == 0:
            image_l, image_r, image_c = countdown(args['start_wait'], False, False, True, i+1)
        else:
            image_l, image_r, image_c = countdown(args['wait'], False, False, True, i+1)
            
        #same images
        cv2.imwrite(os.path.join(args['center_folder'], f'C_{i:03d}.jpg'), image_c)
        
        
    # Proccess Left Stereo Images
    
    # iterate through all images
    for i in range(args['num']):
        if i == 0:
            image_l, image_r, image_c = countdown(args['start_wait'], True, True, False, i+1)
        else:
            image_l, image_r, image_c = countdown(args['wait'], True, True, False, i+1)

        # save image

        cv2.imwrite(os.path.join(args['stereo_left_folder'], f'L_{i:03d}.jpg'), image_l)
        cv2.imwrite(os.path.join(args['stereo_left_folder'], f'C_{i:03d}.jpg'), image_c)
        
    # Proccess Right Stereo Images
    
    for i in range(args['num']):
        if i == 0:
            image_l, image_r, image_c = countdown(args['start_wait'], True, False, False,  i+1)
        else:
            image_l, image_r, image_c = countdown(args['wait'], True, False, False, i+1)

        # save image
        cv2.imwrite(os.path.join(args['stereo_right_folder'], f'R_{i:03d}.jpg'), image_r)
        cv2.imwrite(os.path.join(args['stereo_right_folder'], f'C_{i:03d}.jpg'), image_c)
    

    # When everything done, release the capture
    left.release()
    right.release()
    center.release() 
    cv2.destroyAllWindows()
    
    
    # Calculate Calibration Parameters
    # set up board
    board_size = (args['width'], args['height'])
    idxs = np.zeros((board_size[0]*board_size[1],3), np.float32)
    idxs[:,:2] = np.mgrid[0:board_size[0],0:board_size[1]].T.reshape(-1,2)
    idxs *= args['distance']
    
    # find left params
    print("\nStarting left camera...")
    files = [os.path.join(args['left_folder'], f) for f in os.listdir(args['left_folder'])]
    mtx_l, dist_l, _ = find_params(files, idxs)

    # find right params
    print("\nStarting right camera...")
    files = [os.path.join(args['right_folder'], f) for f in os.listdir(args['right_folder'])]
    mtx_r, dist_r, _ = find_params(files, idxs)
    
    # find center params
    print("\nStarting center camera...")
    files = [os.path.join(args['center_folder'], f) for f in os.listdir(args['center_folder'])]
    mtx_c, dist_c, _ = find_params(files, idxs)


    # get all imgpts of stereo imgs
    print("\nStarting stereo calibration...")
    files = [os.path.join(args['stereo_left_folder'], fl) for fl in os.listdir(args['stereo_left_folder'])]
    files_l = sorted([fl for fl in files if 'L' in fl])
    files_cl = sorted([fl for fl in files if 'C' in fl])
    _, _, imgpts_l = find_params(files_l, idxs, output=False)
    _, _, imgpts_cl = find_params(files_cl, idxs, output=False)
    
    files = [os.path.join(args['stereo_right_folder'], fr) for fr in os.listdir(args['stereo_right_folder'])]
    files_cr = sorted([fr for fr in files if 'C' in fr])
    files_r = sorted([fr for fr in files if 'R' in fr])
    _, _, imgpts_cr = find_params(files_cr, idxs, output=False)
    _, _, imgpts_r = find_params(files_r, idxs, output=False)

    # stereo left calibration
    real_pts = [idxs]*len(imgpts_l)
    shape = cv2.imread(files[0]).shape[::-1][1:]
    ret, _, _, _, _, R_L, T_L, E_L, F_L = cv2.stereoCalibrate(real_pts, imgpts_l, imgpts_cl, 
                                                        mtx_l, dist_l, mtx_c, dist_c, shape,
                                                        criteria=criteria, flags=cv2.CALIB_FIX_INTRINSIC)
    print("Stereo Left Calibration")
    print("Rotation:")
    print(R_L)
    print("\nTranslation: ")
    print(T_L)
    print("\nEssential: ")
    print(E_L)
    print("\nFundamental: ")
    print(F_L)    

    # get rectification params
    R_l, R_cl, P_l, P_cl, Q_L, _, _ = cv2.stereoRectify(mtx_l, dist_l, mtx_c, dist_c, shape, R_L, T_L)

    # save params
    np.savez(args['outfile_left'], mtx_l=mtx_l, dist_l=dist_l, mtx_c=mtx_c, dist_c=dist_c, R_L=R_L, T_L=T_L, E_L=E_L, F_L=F_L, Q_L=Q_L, R_l=R_l, P_l=P_l, R_cl=R_cl, P_cl=P_cl)
    
    # stereo right calibration
    real_pts = [idxs]*len(imgpts_r)
    shape = cv2.imread(files[0]).shape[::-1][1:]
    ret, _, _, _, _, R_R, T_R, E_R, F_R = cv2.stereoCalibrate(real_pts, imgpts_cr, imgpts_r, 
                                                        mtx_c, dist_c, mtx_r, dist_r, shape,
                                                        criteria=criteria, flags=cv2.CALIB_FIX_INTRINSIC)
    print("Stereo Right Calibration")
    print("Rotation:")
    print(R_R)
    print("\nTranslation: ")
    print(T_R)
    print("\nEssential: ")
    print(E_R)
    print("\nFundamental: ")
    print(F_R)    

    # get rectification params
    R_cr, R_r, P_cr, P_r, Q_R, _, _ = cv2.stereoRectify(mtx_c, dist_c, mtx_r, dist_r, shape, R_R, T_R)

    # save params
    np.savez(args['outfile_right'], mtx_c=mtx_c, dist_c=dist_c, mtx_r=mtx_r, dist_r=dist_r, R_R=R_R, T_R=T_R, E_R=E_R, F_R=F_R, Q_R=Q_R, R_cr=R_cr, P_cr=P_cr, R_r=R_r, P_r=P_r)
    
