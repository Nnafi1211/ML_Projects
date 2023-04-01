import cv2
import mediapipe as mp
import time

class FaceDetector():
    
    def __init__(self, minDitectionCon=0.5):
        
        self.minDitectionCon = minDitectionCon
        self.mpFace = mp.solutions.face_detection
        self.mpDraw = mp.solutions.drawing_utils
        self.face = self.mpFace.FaceDetection(minDitectionCon)


    def findFaces(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.face.process(imgRGB)
        
        bboxes = []
        
        if self.results.detections:
            for id, detection in enumerate(self.results.detections):
                
                bboxC = detection.location_data.relative_bounding_box 
                #bouding box coming from class
                h, w, c = img.shape
                bbox = int(bboxC.xmin * w), int(bboxC.ymin * h), \
                        int(bboxC.width * w), int(bboxC.height * h)
                
                bboxes.append([id, bbox, detection.score])
                
                img = self.fancyDraw(img, bbox)
                
                # cv2.rectangle(img, bbox, (255, 0, 255), 2)
                cv2.putText(img, f'{int(detection.score[0]*100)}%', (bbox[0], bbox[1]-20), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)

        return img, bboxes

    def fancyDraw(self, img, bbox, length=30, thickness=5, rect_thickness=1):
        x, y, w, h = bbox
        x1, y1 = x + w, y + h
        
        cv2.rectangle(img, bbox, (255, 0, 255), rect_thickness)
        cv2.line(img, (x, y), (x+length, y), (255, 0, 255), thickness)
        cv2.line(img, (x, y), (x, y+length), (255, 0, 255), thickness)
        
        cv2.line(img, (x, y1), (x+length, y1), (255, 0, 255), thickness)
        cv2.line(img, (x, y1), (x, y1-length), (255, 0, 255), thickness)
        
        cv2.line(img, (x1, y), (x1-length, y), (255, 0, 255), thickness)
        cv2.line(img, (x1, y), (x1, y+length), (255, 0, 255), thickness)
        
        cv2.line(img, (x1, y1), (x1-length, y1), (255, 0, 255), thickness)
        cv2.line(img, (x1, y1), (x1, y1-length), (255, 0, 255), thickness)
        
        return img

def main():
    cap = cv2.VideoCapture(0)
    pTime = 0   
    
    detector = FaceDetector() 
    
    while True:
        success, img = cap.read()
        img, bboxes = detector.findFaces(img)
        
        print(bboxes)
        
        cTime = time.time()
        fps = 1 / (cTime-pTime)
        pTime = cTime
        
        cv2.putText(img, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 2)
        cv2.imshow("Face detection", img)
        cv2.waitKey(25)

if __name__ == '__main__':
    main()