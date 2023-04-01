import cv2
import mediapipe
import time
import numpy as np
import HandTrackingModule as htm

cap = cv2.VideoCapture(2)

detector = htm.HandDetector(detectionCon=0.85)

#############
colors = [(255, 255, 68), (0, 255, 0), (0, 0, 255), (255, 255, 255)]
selectedColor = 0
chageColor = True
xp, yp = 0, 0
imgCanvas = np.zeros((480, 640, 3), np.uint8)
#############

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1) # Flipping is important for painting
    img = detector.findHands(img, draw=True)
    # Extracting the Hand Position
    lmList = detector.findPosition(img)
    
    if len(lmList):
        x1, y1 = lmList[8][1:] # Tip of the index finger
        x2, y2 = lmList[12][1:] # Tip of the middle finger
        fingers = detector.fingersUp()
        
        if fingers[1] and sum(fingers)==1:
            # print('Drawing Mode')
            cv2.circle(img, (x1, y1), 15, colors[selectedColor], cv2.FILLED)
            
            if xp==0 and yp==0:
                xp, yp = x1, y1
            cv2.line(imgCanvas, (xp, yp), (x1, y1), colors[selectedColor], 5)
            xp, yp = x1, y1
            
        if fingers[1] and fingers[2] and sum(fingers)==2:
            # print('Relax Mode')
            xp, yp = 0, 0
            
        if fingers[1] and fingers[2] and fingers[0] and sum(fingers)==3:
            # print('Color Change')
            if chageColor:
                selectedColor = (selectedColor + 1) % 3
                chageColor =  False
                
        if not fingers[0]:
            chageColor = True
            
        if sum(fingers)==5:
            # print('Erase Mode')
            cv2.line(imgCanvas, (xp, yp), (x1, y1), (0, 0, 0), 25)
            cv2.circle(img, (x1, y1), 25, (255, 255, 255), cv2.FILLED)
    
    
    imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img, imgInv)
    img = cv2.bitwise_or(img, imgCanvas)
            
    
    cv2.rectangle(img, (0, 0), (640, 10), colors[selectedColor], cv2.FILLED)
    cv2.imshow('Magical Painter', img)
    # cv2.imshow('Canvas', imgCanvas)
    cv2.waitKey(20)