import cv2
import mediapipe as mp
import time
import os
from modules import HandTrackingModule as htm

cap = cv2.VideoCapture(2)
pTime = 0

detector = htm.HandDetector(detectionCon=0.65)
fTips = [4, 8, 12, 16, 20]

totalFingers = 0

folderPath = 'Numbers'
myList = os.listdir(folderPath)
# print(myList)
numberList = []
for impath in myList:
    image = cv2.imread(f'{folderPath}/{impath}')
    numberList.append(image)

while True:
    success, img = cap.read()
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)
    
    if len(lmList):
        fingers = []
            
        for fTip in fTips:
            if fTip==4:
                # For Left Hand
                if lmList[4][1] < lmList[20][1]: 
                    if lmList[fTip][1] < lmList[fTip-1][1]:
                        fingers.append(1)
                    else:
                        fingers.append(0)
                # For Right Hand
                else: 
                    if lmList[fTip][1] > lmList[fTip-1][1]:
                        fingers.append(1)
                    else:
                        fingers.append(0)
            else:
                if lmList[fTip][2] < lmList[fTip-2][2]:
                    fingers.append(1)
                else:
                    fingers.append(0)
        
        print(fingers)
        totalFingers = sum(fingers)
    
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    
    
    cv2.putText(img, f'Total Fingers: {totalFingers}', (400, 70), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0,165,255), 2)
    cv2.putText(img, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0,165,255), 3)
    cv2.imshow('Video', img)
    cv2.waitKey(1)
    