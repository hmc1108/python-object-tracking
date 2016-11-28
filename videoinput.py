# -*- coding: utf-8 -*-

'''
It can read from a single image, or a sequence of them,
or a video, or finally a camera. You can specify all
these types of sources in the initialization string, as well as a rescaling
(rows x cols), or if you want the image source to cycle (return
at the end), or directly terminate (in which case the method
Read will simply return None).

Note 1: If the initialization parameters are not set (or are set, but 0) 
then the input image is not scaled, and
Is provided according to its original size.

Note 2:

Possible examples of valid initialization strings:

Read non-scaled video file, and repeat at end (loop):
"/path/to/file.mp4:loop"

Equivalent to the previous one:
"/path/to/file.mp4:rows=0:cols=0:loop"

Similar, but with another type of container (supports almost any type of
video):
"/path/to/file.avi"

Reading sequence of images by scaling only the width, and loop when finishing:
"/path/to/img-???.jpg:rows=0:cols=200:loop"

Reading a single image. If a second reading is attempted,
will return None (as there is no loop):
"/path/to/img-001.jpg:rows=300:cols=400"

Reading sequence of images, each to its original size, and without loop
(When finished, the read method will return None):
"/path/to/img-*.jpg"

Numbered camera reading as 0:
"0"

Camera reading numbered as 1, with rescaling:
"1: rows = 100: cols = 200"

'''
import glob
import cv2


class VideoInput(object):
    def __init__(self, str):
        largs = map(lambda x: tuple(x.split('=')), str.split(':'))
        self.cols, self.rows = 0, 0
        self.loop = False
        self.type = 'camera'
        self.camera = 0
        for arg in largs:
            if len(arg) == 1:
                if arg[0] == 'loop':
                    self.loop = True
                elif arg[0].lower().endswith(('.jpg', '.jpeg',
                                              '.png', '.gif')):
                    self.type = 'imgfiles'
                    self.imgfiles = glob.glob(arg[0])
                    self.imgfiles.sort()
                    self.curframe = 0
                elif arg[0].lower().endswith(('.mpeg', '.mpg', '.dv', '.wmv',
                                              '.avi', '.mp4', '.webm', '.mkv')):
                    self.type = 'videofile'
                    self.videofile = arg[0]
                    self.cap = cv2.VideoCapture(self.videofile)
                elif arg[0].isdigit():
                    self.type = 'camera'
                    self.camera = int(arg[0])
                    self.cap = cv2.VideoCapture(self.camera)
            else:
                var, val = arg
                if(var == 'cols'):
                    self.cols = int(val)
                elif(var == 'rows'):
                    self.rows = int(val)
        if self.cols != 0 and self.type == 'camera':
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cols)
        if self.rows != 0 and self.type == 'camera':
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.rows)

    def read(self):
        if self.type in ('camera', 'videofile'):
            flag, frame = self.cap.read()
            if self.loop and frame is None:
                self.cap.set(cv2.CAP_PROP_POS_AVI_RATIO, 0)
                flag, frame = self.cap.read()
        elif self.type in ('imgfiles', ):
            if self.curframe == len(self.imgfiles):
                if self.loop:
                    self.curframe = 0
                else:
                    return None
            frame = cv2.imread(self.imgfiles[self.curframe])
            self.curframe += 1
        if frame is not None:  # Posible escalado
            if self.cols != 0 and self.rows != 0:
                frame = cv2.resize(frame, (self.cols, self.rows))
            elif self.cols != 0:
                frame = cv2.resize(frame, (self.cols, frame.shape[0]))
            elif self.rows != 0:
                frame = cv2.resize(frame, (frame.shape[1], self.rows))
        return frame

    def close(self):
        if self.type in ('videofile', 'camera'):
            # try:
                self.cap.release()
