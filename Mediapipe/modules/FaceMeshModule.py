import cv2
import mediapipe as mp
import time

class FaceMeshDetector():
    
    def __init__(self, 
                static_image_mode=False,
                max_num_faces=1,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5):
        self.static_image_mode = static_image_mode
        self.max_num_faces = max_num_faces
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.mpDraw = mp.solutions.drawing_utils
        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(static_image_mode,
                                                      max_num_faces,
                                                      min_detection_confidence,
                                                      min_tracking_confidence)
        self.drawSpec = self.mpDraw.DrawingSpec(thickness=1, circle_radius=2)


    def findFaceMesh(self, img, draw=True):
        
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        self.results = self.faceMesh.process(imgRGB)

        faces = []
        if self.results.multi_face_landmarks:
            for faceLms in self.results.multi_face_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, faceLms, self.mpFaceMesh.FACE_CONNECTIONS, self.drawSpec, self.drawSpec)

                face = []
                for id, lm in enumerate(faceLms.landmark):
                    ih, iw, ic = img.shape
                    cx, cy = int(lm.x*iw), int(lm.y*ih)
                    face.append([cx, cy])
                    # print(id, cx, cy)
                
                faces.append(face)
        
        return img, faces
    
    
def main():
    cap = cv2.VideoCapture(0)
    pTime = 0
    
    detector = FaceMeshDetector(max_num_faces=2)
    
    while True:
        success, img = cap.read()
        img, faces = detector.findFaceMesh(img)
        
        if len(faces) != 0:
            print(len(faces))
        
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        
        cv2.putText(img, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
        
        cv2.imshow("Face Mesh", img)
        cv2.waitKey(10)
    
if __name__ == '__main__':
    main()