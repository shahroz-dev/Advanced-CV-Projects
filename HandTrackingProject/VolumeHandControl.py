import cv2
import time
import numpy as np
import handTrackingModule as htm
import math
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
import os

########PARAMETERS########
frameWidth = 640
frameHeight = 480
##########################

cap = cv2.VideoCapture(0)
cap.set(3, frameWidth)
cap.set(4, frameHeight)
pTime = 0

detector = htm.handDetector(detectionCon=0.7)

devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
# volume.GetMute()
# volume.GetMasterVolumeLevel()
volRange = volume.GetVolumeRange()
minVolume = volRange[0]
maxVolume = volRange[1]
vol = 0
volBar = 400
volPer = 0

fourcc = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')
dst_path = "Saved_Results/"
if not os.path.exists(dst_path):
    os.makedirs(dst_path)
videoWriter = cv2.VideoWriter(dst_path + 'gestureVolControl.avi', fourcc, 30.0, (640, 480))

while cap.isOpened():
    success, img = cap.read()
    if success:

        img = detector.findHands(img)
        lmList = detector.findPosition(img, draw=False)
        if len(lmList) != 0:
            # print(lmList[4], lmList[8])

            x1, y1 = lmList[4][1], lmList[4][2]
            x2, y2 = lmList[8][1], lmList[8][2]

            # get center of line
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 15, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

            # find the length of line using euclidean function sqrt((x2-x1)^2 + (y2-y1)^2).
            length = math.hypot(x2-x1, y2-y1)
            # print(length)

            # Hande range 25 - 200
            # Volume Range -65 - 0
            vol = np.interp(length, [25, 200], [minVolume, maxVolume])
            volBar = np.interp(length, [25, 200], [400, 150])
            volPer = np.interp(length, [25, 200], [0, 100])
            # print(vol)
            volume.SetMasterVolumeLevel(vol, None)

            if length < 30:
                cv2.circle(img, (cx, cy), 15, (0, 255, 0), cv2.FILLED)

        cv2.rectangle(img, (50, 150), (85, 400), (255, 0, 0), 3)
        cv2.rectangle(img, (50, int(volBar)), (85, 400), (255, 0, 0), cv2.FILLED)
        cv2.putText(img, '{}%'.format(str(int(volPer))), (45, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, 'FPS: {}'.format(str(int(fps))), (20, 40), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)
        cv2.imshow("Image", img)
        videoWriter.write(img)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
    else:
        break

cap.release()
videoWriter.release()
cv2.destroyAllWindows()
