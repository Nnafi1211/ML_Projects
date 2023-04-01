import cv2
import mediapipe as mp
import time

class PoseDetector():
    def __init__(self, 
                mode=False,
                complexity=1,
                smooth=True,
                detection_con=0.5,
                tracking_con=0.5):
        
        self.mode = mode
        self.complexity = complexity
        self.smooth = smooth
        self.detection_con = detection_con
        self.tracking_con = tracking_con
        
        self.mpPose = mp.solutions.pose
        self.mpDraw = mp.solutions.drawing_utils
        self.pose = self.mpPose.Pose(self.mode, self.complexity, 
                                    self.smooth, self.detection_con, 
                                    self.tracking_con)


    def findPose(self, img, draw=True):
        
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)

        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)
        
        return img    
        
        
    def findPosition(self, img, draw=True):
        
        lmList = []
        
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
        
        return lmList
    
    
def main():
    cap = cv2.VideoCapture('vid/5.mp4')

    pTime = 0
    
    detector = PoseDetector()

    while True:
        success, img = cap.read()
        detector.findPose(img)
        lmList = detector.findPosition(img, draw=False)
        
        print(lmList)
        
        if len(lmList) != 0:
            cx = lmList[14][1]
            cy = lmList[14][2]
            
            cv2.circle(img, (cx, cy), 10, (255, 0, 255), cv2.FILLED)
        
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        
        cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
        
        cv2.imshow("Pose Estimaton", img)
        cv2.waitKey(1)

if __name__ == '__main__':
    main()