import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(2)
pTime = 0

mpFace = mp.solutions.face_detection
mpDraw = mp.solutions.drawing_utils
face = mpFace.FaceDetection()

while True:
    success, img = cap.read()
    
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = face.process(imgRGB)
    
    if results.detections:
        for id, detection in enumerate(results.detections):
            # mpDraw.draw_detection(img, detection)
            # print(detection.location_data.relative_bounding_box)
            
            bboxC = detection.location_data.relative_bounding_box 
            #bouding box coming from class
            h, w, c = img.shape
            bbox = int(bboxC.xmin * w), int(bboxC.ymin * h), \
                    int(bboxC.width * w), int(bboxC.height * h)
            cv2.rectangle(img, bbox, (255, 0, 255), 2)
            cv2.putText(img, f'{int(detection.score[0]*100)}%', (bbox[0], bbox[1]-20), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)

    
    cTime = time.time()
    fps = 1 / (cTime-pTime)
    pTime = cTime
    
    cv2.putText(img, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 2)
    cv2.imshow("Face detection", img)
    cv2.waitKey(25)