import time
import cv2
import mediapipe as mp

mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils

frameWidth = 360
cap = cv2.VideoCapture("pose_video.mp4")
pTime = 0
while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(imgRGB)
    # print(results.pose_landmarks)
    if results.pose_landmarks:
        mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
        for id, lm in enumerate(results.pose_landmarks.landmark):
            h, w, c = img.shape
            print(id, lm)
            # get the actual pixel value of landmarks instead of ratio
            cx, cy = int(lm.x*w), int(lm.y*h)
            cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)


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
