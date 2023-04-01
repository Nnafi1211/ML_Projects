import cv2
import mediapipe as mp
import time
import math
from modules import HandTrackingModule as htm
from subprocess import call
import numpy as np

cap = cv2.VideoCapture(2)
pTime = 0

detector = htm.HandDetector(detectionCon=0.65)

call(["amixer", "-D", "pulse", "sset", "Master", "0%"])

minVolume = 0
maxVolume = 100
vol = 0
volBar = 400

while True:
    success, img = cap.read()
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)
    
    if len(lmList):
        # print(lmList[4], lmList[8])
    
        x1, y1 = lmList[4][1], lmList[4][2]
        x2, y2 = lmList[8][1], lmList[8][2]
            
        cv2.circle(img, (x1, y1), 10, (255,105,65), cv2.FILLED)
        cv2.circle(img, (x2, y2), 10, (255,105,65), cv2.FILLED)
        cv2.line(img, (x1, y1), (x2, y2), (255,105,65), 3)
        cx, cy = int((x1+x2)/2), int((y1+y2)/2)
        cv2.circle(img, (cx, cy), 10, (255,105,65), cv2.FILLED)
        
        length = math.hypot(x1-x2, y1-y2)
        if length < 50:
            cv2.circle(img, (cx, cy), 10, (0,0,255), cv2.FILLED)
        if length > 200:
            cv2.circle(img, (cx, cy), 10, (255, 255, 255), cv2.FILLED)
        # print(length)
        
        vol = np.interp(length, [50, 200], [minVolume, maxVolume])
        # print(vol)
        call(["amixer", "-D", "pulse", "sset", "Master", str(vol)+"%"])
    
    volBar = np.interp(vol, [0, 100], [400, 100])
    cv2.rectangle(img, (50, 100), (85, 400), (0,165,255), 3)
    cv2.rectangle(img, (50, int(volBar)), (85, 400), (0,165,255), cv2.FILLED)
    cv2.putText(img, f'{int(vol)}%', (50, 430), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0,165,255), 2)
    
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    
    cv2.putText(img, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0,165,255), 3)
    cv2.imshow("Video", img)
    cv2.waitKey(1)
       