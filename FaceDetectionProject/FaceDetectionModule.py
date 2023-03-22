import time
import cv2
import mediapipe as mp


class FaceDetector:
    def __init__(self, min_det_cf=0.5, model_sel=0):
        self.results = None
        self.min_det_cf = min_det_cf
        self.model_sel = model_sel
        self.mpFaceDetection = mp.solutions.face_detection
        self.mpDraw = mp.solutions.drawing_utils
        self.faceDetection = self.mpFaceDetection.FaceDetection(self.min_det_cf, self.model_sel)

    def findFaces(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceDetection.process(imgRGB)
        bboxs = []
        if self.results.detections:
            for id, detection in enumerate(self.results.detections):
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, ic = img.shape
                bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                       int(bboxC.width * iw), int(bboxC.height * ih)
                bboxs.append([id, bbox, detection.score])
                if draw:
                    self.fancyDraw(img, bbox)
                    cv2.putText(img, '{}%'.format(str(int(detection.score[0] * 100))), (bbox[0], bbox[1] - 20)
                                , cv2.FONT_HERSHEY_PLAIN, 8, (255, 0, 255), 8)
        return img, bboxs

    def fancyDraw(self, img, bbox, l=60, t=20, rt=5):
        x, y, w, h = bbox
        x1, y1 = x + w, y + h  # corner pts at the diagonal positions
        cv2.rectangle(img, bbox, (255, 0, 255), 2, rt)
        # Top left x,y
        cv2.line(img, (x, y), (x + l, y), (255, 0, 255), t)
        cv2.line(img, (x, y), (x, y+l), (255, 0, 255), t)
        # Top right x,y
        cv2.line(img, (x1, y), (x1 - l, y), (255, 0, 255), t)
        cv2.line(img, (x1, y), (x1, y + l), (255, 0, 255), t)
        # bottom left x,y
        cv2.line(img, (x, y1), (x + l, y1), (255, 0, 255), t)
        cv2.line(img, (x, y1), (x, y1 - l), (255, 0, 255), t)
        # Top right x,y
        cv2.line(img, (x1, y1), (x1 - l, y1), (255, 0, 255), t)
        cv2.line(img, (x1, y1), (x1, y1 - l), (255, 0, 255), t)
        return img


def main():
    frameWidth = 360
    cap = cv2.VideoCapture("face_video_3.mp4")
    pTime = 0
    detector = FaceDetector()
    while cap.isOpened():
        success, img = cap.read()
        if success:
            img, bboxs = detector.findFaces(img)
            print(bboxs)

            r = frameWidth / img.shape[1]  # width height ratio
            dim = (frameWidth, int(img.shape[0] * r))
            img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

            cTime = time.time()
            fps = 1 / (cTime - pTime)
            pTime = cTime
            cv2.putText(img, 'FPS: {}'.format(str(int(fps))), (20, 40), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
            cv2.imshow("Video", img)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
        else:
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
