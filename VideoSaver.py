import numpy as np
import cv2
import os

import time

class VideoSaver():
    gShowType = "dump" #play or dump video
    gDumpDir = "./render"
    gDumpFd = None
    gFps = 15

    def __init__(self, showType="dump", dumpDir="./render", width=640, height=480, fps=15):
        if showType not in ["dump", "play"]:
            print("demoShow.init Failed, Invalid showType:%s" %(showType))
            return -1

        self.gShowType = showType
        self.gDumpDir = dumpDir
        self.gFps = fps
        if self.gShowType == "dump":
            if not os.path.isdir(self.gDumpDir):
                os.mkdir(self.gDumpDir)
            video_name = self.gDumpDir + "/" + time.strftime("%Y%m%d_%H%M%S", time.localtime()) + '.avi'
            if os.path.isfile(video_name):
                os.remove(video_name)

            (major, _, _) = cv2.__version__.split(".")
            if major == '2':
                fourcc = cv2.cv.CV_FOURCC('m', 'p', '4', 'v')
            else:
                fourcc = cv2.VideoWriter_fourcc(*'MJPG')

            print("demoShow.init openning %s " %(video_name))
            self.gDumpFd = cv2.VideoWriter(video_name, fourcc, fps, (width, height))
            if self.gDumpFd == None:
                print("demoShow.init Failed, open %s failed" %(video_name))
                return -1
        print("demoShow.init success")

    def __del__(self):
        if self.gDumpFd != None:
            self.gDumpFd.release()
            self.gDumpFd = None

    def addFrame(self, img):
        if self.gShowType == "dump":
            if self.gDumpFd == None:
                print("demoShow.addFrame failed, Please call init success first")
                return -1
            self.gDumpFd.write(img)
        else:
            sleepTime = 1000.0/self.gFps
            cv2.imshow('image', img)
            cv2.waitKey(sleepTime)