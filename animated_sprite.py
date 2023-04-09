# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import cv2
import glob
from threading import Thread
import time


class Frame:
    def __init__(self, frame=None, nxt=None):
        self.frame = frame
        self.nextFrame = nxt
    def next(self):
        return self.nextFrame

class AnimatedSprite:
    
    def __init__(self,images, fps=30, resize=None):
        imList = images
        if type(images)==str:
            if resize:
                imList = [cv2.resize(cv2.imread(img,cv2.IMREAD_UNCHANGED), dsize=resize, interpolation=cv2.INTER_CUBIC) for img in glob.glob(images)]
            else:
                imList = [cv2.imread(img,cv2.IMREAD_UNCHANGED) for img in glob.glob(images)]

        curFrame = Frame()
        self.firstFrame = curFrame
        for image in imList[:-1]:
            curFrame.frame = image
            nextFrame = Frame()
            curFrame.nextFrame = nextFrame
            curFrame = nextFrame
        curFrame.frame=imList[-1]
        self.lastFrame = curFrame
        self.lastFrame.nextFrame = self.firstFrame
        self.curFrame = self.lastFrame
        self.time = 1/fps
        self.animThread = Thread(target = self.clk)
        self.animThread.setDaemon(True)
    def next(self):
        self.curFrame=self.curFrame.next()
    
    def out(self):
        return self.curFrame.frame.copy()
    
    def start(self):
        self.animThread.start()
    
    def clk(self):
        while(1):
            time.sleep(self.time)
            self.next()
    
    def reset(self):
        self.curFrame = self.firstFrame

if __name__ == '__main__':
    a = AnimatedSprite('sprites/sample/*.png')
    a.start()
    alist = [cv2.imread(img) for img in glob.glob('sprites/sample/*.png')]
    while(1):
        cv2.imshow('frame', a.out())
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


