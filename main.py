# -*- coding: utf-8 -*-

import sys
import cv2
import time

import videoinput
import utilscv
import objrecogn as orec

if __name__ == '__main__':

    # Creating window and associated sliders, and mouse callback
    def nothing(*arg):
        pass
    cv2.namedWindow('Features')
    cv2.namedWindow('ImageDetector')

    # Reprojection error to calculating inliers with RANSAC
    cv2.createTrackbar('projer', 'Features', 5, 10, nothing)
    # Number of minimum inliers to indicate that an object has been recognized
    cv2.createTrackbar('inliers', 'Features', 20, 50, nothing)
    # Draw keypoints in frame extracted from video
    cv2.createTrackbar('drawKP', 'Features', 0, 1, nothing)

    # Video source opening
    if len(sys.argv) > 1:
        strsource = sys.argv[1]
    else:
        strsource = '0:rows=300:cols=400'  # Simple aperture of webcam, no scaling
    videoinput = videoinput.VideoInput(strsource)
    paused = False
    methodstr = 'None'

    dataBaseDictionary = orec.loadModelsFromDirectory()
    passKey = 0 # for pass several frames
    while True:
        # Reading input frame, and interface parameters
        if not paused:
            frame = videoinput.read()

        if frame is None:
            print('End of video input')
            break

        if passKey != 3:
            cv2.imshow('Features', frame)
            passKey += 1
            continue
        else:
            passKey = 0



        detector = cv2.xfeatures2d.SURF_create(800)
        # Make GRAY frame
        imgin = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # The output frame
        imgout = frame.copy()
        # Detect features, measure time
        t1 = time.time()
        kp, desc = detector.detectAndCompute(imgin, None)
        if desc is None:
            cv2.imshow('Features', imgout)
            continue
        if len(dataBaseDictionary) > 0:
            #Perform mutual matching
            imgsMatchingMutual = orec.findMatchingMutualOptimize(dataBaseDictionary, desc, kp)
            minInliers = int(cv2.getTrackbarPos('inliers', 'Features'))
            projer = float(cv2.getTrackbarPos('projer', 'Features'))
            # The best image is calculated based on the number of inliers.
            # The best image is one that has more number of inliers, but always
            # exceeding the minimum indicated in the trackbar 'minInliers'
            bestImage, inliersWebCam, inliersDataBase = orec.calculateBestImageByNumInliers(dataBaseDictionary, projer, minInliers)
            if not bestImage is None:
                #If we find a good image, we calculate the affinity matrix and draw the recognized object on the screen.
                orec.calculateAffinityMatrixAndDraw(bestImage, inliersDataBase, inliersWebCam, imgout)
               
        t1 = 1000 * (time.time() - t1)  # Time in milliseconds
        # Get dimension of descriptors for each feature:
        if desc is not None:
            if len(desc) > 0:
                dim = len(desc[0])
            else:
                dim = -1
        # Draw features, and write informative text about the image
        if (int(cv2.getTrackbarPos('drawKP', 'Features')) > 0):
            cv2.drawKeypoints(imgout, kp, imgout,
                              flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        utilscv.draw_str(imgout, (20, 20),
                         "{0} features found, desc. dim. = {1} ".
                         format(len(kp), dim))
        utilscv.draw_str(imgout, (20, 40), "Time (ms): {0}".format(str(t1)))
        # Show results and check keys:
        cv2.imshow('Features', imgout)
        ch = cv2.waitKey(5) & 0xFF
        if ch == 27:  #End when press 'Escape'
            break
        elif ch == ord(' '):  # Pause when press 'Spacebar'
            paused = not paused
        elif ch == ord('.'):  # Point advances single frame
            paused = True
            frame = videoinput.read()

    # Close window (s) and video source (s):
    videoinput.close()
    cv2.destroyAllWindows()
