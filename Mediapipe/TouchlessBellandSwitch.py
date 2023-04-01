import cv2
import mediapipe as mp
from playsound import playsound
import serial
import time
import math
from modules import HandTrackingModule as htm

cap = cv2.VideoCapture(2)

detector = htm.HandDetector(detectionCon=0.85)

serialcomm = serial.Serial('/dev/ttyACM0', 9600) # OP pin 2
serialcomm.timeout = 1

pTime = 0
cTime = 0
dis = 0
flag = 0
mflag = 0
x1, x2, y1, y2 = 0,0,0,0

while True:
    success, img = cap.read()
    img = detector.findHands(img, draw=True)
    # Extracting the Hand Position
    lmList = detector.findPosition(img)
    
    if len(lmList):
        x1, y1 = lmList[8][1:] # Tip of the index finger
        x2, y2 = lmList[4][1:] # Tip of the middle finger
        fingers = detector.fingersUp()
        
        dis = math.sqrt((x1-x2)**2 + (y1-y2)**2)
        
        if dis <= 20:
            flag = 1
            if mflag == 0:
                mflag = 1
        else:
            mflag = 0
            flag = 0
        
        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime
        
        cv2.putText(img, "FPS: "+str(int(fps)), (20, 70), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)
        if flag==0:
            pass
        else:
            if mflag == 1:
                mflag = 2
                playsound('beep-01a.mp3')
            
        if fingers[1] and fingers[2] and sum(fingers)==2:
            cv2.putText(img, "Switch: ON", (440, 70), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)
            e = '\n'
            serialcomm.write(e.encode())
            serialcomm.write('1'.encode())
        else: 
            cv2.putText(img, "Switch: OFF", (440, 70), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)
            e = '\n'
            serialcomm.write(e.encode())
            serialcomm.write('0'.encode())
            
    cv2.imshow('Touchless Bell and Switch', img)
    cv2.waitKey(20)