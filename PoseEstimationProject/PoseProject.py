import cv2
import time
import PoseModule as pm


frameWidth = 360
cap = cv2.VideoCapture("pose_video.mp4")
pTime = 0

detector = pm.poseDetector()

while True:
    success, img = cap.read()
    img = detector.findPose(img)
    lm_list = detector.findPosition(img, draw=False)
    if len(lm_list) != 0:
        print(lm_list[14])
        cv2.circle(img, (lm_list[14][1], lm_list[14][2]), 15, (0, 0, 255), cv2.FILLED)
        cv2.putText(img, str(lm_list[14][0]), (lm_list[14][1], lm_list[14][2]), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

    r = frameWidth / img.shape[1]  # width height ratio
    dim = (frameWidth, int(img.shape[0] * r))
    img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

    cv2.imshow("Video", img)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or not success:
        break

cap.release()
cv2.destroyAllWindows()