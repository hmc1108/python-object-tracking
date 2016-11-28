# -*- coding: utf-8 -*-

import cv2
import os
import numpy as np
import utilscv

# This class will contain for each of the images in the 'sample' folder,
# information needed to compute object recognition.
class ImageFeature(object):

    def __init__(self, nameFile, shape, imageBinary, kp, desc):
        # File name
        self.nameFile = nameFile
        # Shape of the image
        self.shape = shape
        # Binary data of the image
        self.imageBinary = imageBinary
        #Keypoints of the image once applied the feature detection algorithm
        self.kp = kp
        #Descriptors of the features detected
        self.desc = desc
        #Matchings of the image of the database with the image of the webcam
        self.matchingWebcam = []
        #Matching the webcam with the current image of the database.
        self.matchingDatabase = []

    #Clear matching for a new image
    def clearMatchingMutual(self):
        self.matchingWebcam = []
        self.matchingDatabase = []

# Function is responsible for calculating
# the features of each image is in the directory "sample"
def loadModelsFromDirectory():
    # The method return the list of objects typed ImageFeature, 
    # contain all data of the features of images in folder
    dataBase = []
    surf = cv2.xfeatures2d.SURF_create(400)
    for imageFile in os.listdir("sample"):
        #Read image
        colorImage = cv2.imread("sample/" + str(imageFile))
        #Make Gray iamge
        currentImage = cv2.cvtColor(colorImage, cv2.COLOR_BGR2GRAY)
        #Detect and Compute features in image
        kp, desc = surf.detectAndCompute(currentImage, None)
        dataBase.append(ImageFeature(imageFile, currentImage.shape, colorImage, kp, desc))
    return dataBase
    
# Function is responsible for calculating mutual matching, but nesting loops
# It is a very slow solution because it does not take advantage of Numpy library
# we do not even put a slider to use this method as it is very slow
def findMatchingMutual(selectedDataBase, desc, kp):
    for imgFeatures in selectedDataBase:
        imgFeatures.clearMatchingMutual()
        for i in range(len(desc)):
            primerMatching = None
            canditatoDataBase = None
            matchingSegundo = None
            candidateWebCam = None
            for j in range(len(imgFeatures.desc)):
                valorMatching = np.linalg.norm(desc[i] - imgFeatures.desc[j])
                if (primerMatching is None or valorMatching < primerMatching):
                    primerMatching = valorMatching
                    canditatoDataBase = j
            for k in range(len(desc)):
                valorMatching = np.linalg.norm(imgFeatures.desc[canditatoDataBase] - desc[k])
                if (matchingSegundo is None or valorMatching < matchingSegundo):
                    matchingSegundo = valorMatching
                    candidateWebCam = k
            if not candidateWebCam is None and i == candidateWebCam:
                imgFeatures.matchingWebcam.append(kp[i].pt)
                imgFeatures.matchingDatabase.append(imgFeatures.kp[canditatoDataBase].pt)
    return selectedDataBase

# Function is responsible for calculating the mutual matching of a webcam image,
# with all the  images in the database.
def findMatchingMutualOptimize(selectedDataBase, desc, kp):
    #The algorithm is repeated for each image in the database.
    for img in selectedDataBase:
        img.clearMatchingMutual()
        for i in range(len(desc)):
             # Calculates the standard difference of the current descriptor, with all
             # descriptors of image in the database. Using Numpy library, all distance
             # between the current descriptor and all descriptors of the current image
             distanceListFromWebCam = np.linalg.norm(desc[i] - img.desc, axis=-1)
             # The candidate who is the shortest distance from the current descriptor is obtained
             candidatoDataBase = distanceListFromWebCam.argmin() 
             # If the matching is mutual, 
             # it is verified that the candidateDatabase
             # has the current descriptor as best matching
             distanceListFromDataBase = np.linalg.norm(img.desc[candidatoDataBase] - desc,
                                           axis=-1)
             candidatoWebCam = distanceListFromDataBase.argmin()
             #If mutual matching is fulfilled, it is stored for later processing
             if (i == candidatoWebCam):
                img.matchingWebcam.append(kp[i].pt)
                img.matchingDatabase.append(img.kp[candidatoDataBase].pt)

        img.matchingWebcam = np.array(img.matchingWebcam)
        img.matchingDatabase = np.array(img.matchingDatabase)
    return selectedDataBase

# This function calculates the best image based on the number of inliers
# that each image in the database has with the image obtained from the webcam
def calculateBestImageByNumInliers(selectedDataBase, projer, minInliers):
    if minInliers < 15:
        minInliers = 15
    bestIndex = None
    bestMask = None
    numInliers = 0
    # For each image
    for index, imgWithMatching in enumerate(selectedDataBase):
        #RANSAC algorithm is used to calculate the number of inliers
        _, mask = cv2.findHomography(imgWithMatching.matchingDatabase, 
                                     imgWithMatching.matchingWebcam, cv2.RANSAC, projer)
        cv2.findHomography
        if not mask is None:
            # If the number of inliers exceeds the minimum number of inliers,,
            # considering the image that matches the object stored in the database.
            countNonZero = np.count_nonzero(mask)
            if (countNonZero >= minInliers and countNonZero > numInliers):
                numInliers = countNonZero
                bestIndex = index
                bestMask = (mask >= 1).reshape(-1)
    # If you have obtained an image as the best image and, therefore,
    # must have a minimum number of inlers, it is finally calculated
    # the keypoints that are inliers from the mask obtained in findHomography
    if not bestIndex is None:
        bestImage = selectedDataBase[bestIndex]
        inliersWebCam = bestImage.matchingWebcam[bestMask]
        inliersDataBase = bestImage.matchingDatabase[bestMask]
        return bestImage, inliersWebCam, inliersDataBase
    return None, None, None
                
# This function calculates the affinity matrix A, paints a rectangle around
# detected object and paints in a new window the image of the database
# corresponding to the recognized object.
def calculateAffinityMatrixAndDraw(bestImage, inliersDataBase, inliersWebCam, imgout):
    #The affinity matrix A is calculated
    A = cv2.estimateRigidTransform(inliersDataBase, inliersWebCam, fullAffine=True)
    A = np.vstack((A, [0, 0, 1]))
    
    #Calculate the rectangepoints
    a = np.array([0, 0, 1], np.float)
    b = np.array([bestImage.shape[1], 0, 1], np.float)
    c = np.array([bestImage.shape[1], bestImage.shape[0], 1], np.float)
    d = np.array([0, bestImage.shape[0], 1], np.float)
    centre = np.array([float(bestImage.shape[0])/2, 
       float(bestImage.shape[1])/2, 1], np.float)
       
    #The virtual points are multiplied, to convert them into
    #real image points
    a = np.dot(A, a)
    b = np.dot(A, b)
    c = np.dot(A, c)
    d = np.dot(A, d)
    centre = np.dot(A, centre)
    
    # real points
    areal = (int(a[0]/a[2]), int(a[1]/b[2]))
    breal = (int(b[0]/b[2]), int(b[1]/b[2]))
    creal = (int(c[0]/c[2]), int(c[1]/c[2]))
    dreal = (int(d[0]/d[2]), int(d[1]/d[2]))
    centrereal = (int(centre[0]/centre[2]), int(centre[1]/centre[2]))
    
    #Draw the polygon and the file name of the image in the center of the polygon
    points = np.array([areal, breal, creal, dreal], np.int32)
    cv2.polylines(imgout, np.int32([points]),1, (255,255,255), thickness=2)
    utilscv.draw_str(imgout, centrereal, bestImage.nameFile.upper())
    #The detected object is displayed in a separate window
    cv2.imshow('ImageDetector', bestImage.imageBinary)
