import time
import cv2
import mediapipe as mp

frameWidth = 360
cap = cv2.VideoCapture("face_video_3.mp4")
pTime = 0

mpFaceDetection = mp.solutions.face_detection
mpDraw = mp.solutions.drawing_utils
faceDetection = mpFaceDetection.FaceDetection(0.75)

while cap.isOpened():
    success, img = cap.read()
    if success:
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = faceDetection.process(imgRGB)
        # print(results)
        if results.detections:
            for id, detection in enumerate(results.detections):
                # print(id, detection)
                # print(detection.score)
                # print(detection.location_data.relative_bounding_box) # we will get the bounding box (x, y, w, h)
                # mpDraw.draw_detection(img, detection)
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, ic = img.shape
                bbox = int(bboxC.xmin*iw), int(bboxC.ymin*ih), \
                       int(bboxC.width * iw), int(bboxC.height * ih)
                cv2.rectangle(img, bbox, (255, 0, 255), 2)
                cv2.putText(img, '{}%'.format(str(int(detection.score[0]*100))), (bbox[0], bbox[1]-20)
                            , cv2.FONT_HERSHEY_PLAIN, 8, (255, 0, 255), 8)

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