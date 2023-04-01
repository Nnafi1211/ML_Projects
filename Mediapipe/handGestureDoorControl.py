import cv2
import mediapipe as mp
from playsound import playsound
import serial
import time
import math

cap = cv2.VideoCapture(2)

mpHands = mp.solutions.hands
hands = mpHands.Hands(min_tracking_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

serialcomm = serial.Serial('/dev/ttyACM0', 9600)
serialcomm.timeout = 1

pTime = 0
cTime = 0
dis = 0
flag = 0
mflag = 0
x1, x2, y1, y2 = 0,0,0,0

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)  
    
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                print(handLms.landmarks[4])
                if id==4:
                    x1, y1 = cx, cy
                if id==8:
                    x2, y2 = cx, cy
                    
                dis = math.sqrt((x1-x2)**2 + (y1-y2)**2)
                
                if dis <= 20:
                    flag = 1
                    if mflag == 0:
                        mflag = 1
                else:
                    mflag = 0
                    flag = 0
                                
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)
    
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    
    e = '\n'
    serialcomm.write(e.encode())
    serialcomm.write(str(flag).encode())
    
    cv2.putText(img, str(int(fps)), (20, 70), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)
    if flag==0:
        cv2.putText(img, "Door: Open", (440, 70), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)
    else:
        cv2.putText(img, "Door: Close", (440, 70), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)
        if mflag == 1:
            mflag = 2
            playsound('beep-01a.mp3')
    
    cv2.imshow("Image", img)
    cv2.waitKey(1)