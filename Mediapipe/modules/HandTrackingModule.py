import cv2
import mediapipe as mp
import time

class HandDetector():
    def __init__(self, mode=False, complexity=1, maxHands=2, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.complexity = complexity
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.complexity, self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils
        
        self.fTips = [4, 8, 12, 16, 20]
    
    def findHands(self, img, draw=True):
        
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
         
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        
        return img
    
    
    def findPosition(self, img, handNo=0, draw=False):
        
        self.lmList = []
        
        if self.results.multi_hand_landmarks:
            
            targetHand = self.results.multi_hand_landmarks[handNo]
            
            for id, lm in enumerate(targetHand.landmark):
                        h, w, c = img.shape
                        cx, cy = int(lm.x*w), int(lm.y*h)
                        # print(id, cx, cy)
                        self.lmList.append([id, cx, cy])
                        if draw:
                            cv2.circle(img, (cx, cy), 15, (128, 128, 128), cv2.FILLED)

        return self.lmList
    
    def fingersUp(self):
        fingers = []
        
        for fTip in self.fTips:
            
            if fTip == 4:
                # For Left Hand
                if self.lmList[4][1] < self.lmList[20][1]: 
                    if self.lmList[fTip][1] < self.lmList[fTip-1][1]:
                        fingers.append(1)
                    else:
                        fingers.append(0)
                # For Right Hand
                else: 
                    if self.lmList[fTip][1] > self.lmList[fTip-1][1]:
                        fingers.append(1)
                    else:
                        fingers.append(0)
            else:
                if self.lmList[fTip][2] < self.lmList[fTip-2][2]:
                    fingers.append(1)
                else:
                    fingers.append(0)
        
        return fingers


def main():
    cap = cv2.VideoCapture(0)

    pTime = 0
        
    detector = HandDetector()

    while True:
        success, img = cap.read()
        detector.findHands(img)
        lmList = detector.findPosition(img)
        if len(lmList)!=0:
            print(lmList[0])
        
        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime
        
        cv2.putText(img, str(int(fps)), (20, 70), cv2.FONT_HERSHEY_TRIPLEX, 2, (255, 0, 255), 3)
        
        cv2.imshow("Image", img)
        cv2.waitKey(1)
    

if __name__ == '__main__':
    main()