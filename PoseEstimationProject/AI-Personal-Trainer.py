import cv2
import time
import numpy as np
import os
import PoseModule as ptm
import math

########PARAMETERS########
frameWidth = 1080
# frameHeight = 720
##########################

cap = cv2.VideoCapture('gym_pose_video.mp4')
pTime = 0
count = 0
flag = False
frame_counter = 0
detector = ptm.poseDetector()

# fourcc = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')
# dst_path = "Saved_Results/"
# if not os.path.exists(dst_path):
#     os.makedirs(dst_path)
# videoWriter = cv2.VideoWriter(dst_path + 'ai-fitness_trainer.avi', fourcc, 30.0, (640, 480))

while cap.isOpened():
    success, img = cap.read()
    frame_counter += 1
    # If the last frame is reached, reset the capture and the frame_counter
    if frame_counter == cap.get(cv2.CAP_PROP_FRAME_COUNT):
        frame_counter = 0  # Or whatever as long as it is the same as next line
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    if success:
        img = detector.findPose(img, draw=False)
        lm_list = detector.findPosition(img, draw=False)
        if len(lm_list) != 0:
            # print(lm_list[11], lm_list[13], lm_list[15])
            x1, y1 = lm_list[11][1], lm_list[11][2]
            x2, y2 = lm_list[13][1], lm_list[13][2]
            x3, y3 = lm_list[15][1], lm_list[15][2]
            cv2.circle(img, (x1, y1), 20, (0, 255, 0), cv2.FILLED)
            cv2.circle(img, (x2, y2), 20, (0, 255, 0), cv2.FILLED)
            cv2.circle(img, (x3, y3), 20, (0, 255, 0), cv2.FILLED)
            cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.line(img, (x2, y2), (x3, y3), (0, 255, 0), 2)
            try:
                m1 = (y2 - y1) / (x2 - x1)
                m2 = (y3 - y2) / (x3 - x2)
            except:
                pass
            angle = math.atan((m1 - m2) / (1 + m1 * m2)) * (180 / math.pi)
            if angle < 0:
                angle = 180 + angle
            print(angle)
            if angle > 80:
                cv2.circle(img, (x1, y1), 20, (255, 0, 0), cv2.FILLED)
                cv2.circle(img, (x2, y2), 20, (255, 0, 0), cv2.FILLED)
                cv2.circle(img, (x3, y3), 20, (255, 0, 0), cv2.FILLED)
                cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.line(img, (x2, y2), (x3, y3), (255, 0, 0), 2)

            if angle <= 80 and not flag:
                flag = True
                count += 1

            if angle >= 150 and flag:
                flag = False

        # cv2.rectangle(img, (20, 700), (250, 1050), (255, 0, 0), cv2.FILLED)
        cv2.putText(img, str(count), (30, 980), cv2.FONT_HERSHEY_COMPLEX, 10, (0, 255, 0), 15, cv2.FILLED)

        r = frameWidth / img.shape[1]  # width height ratio
        dim = (frameWidth, int(img.shape[0] * r))
        img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, 'FPS: {}'.format(str(int(fps))), (20, 40), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Image", img)
        # videoWriter.write(img)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
    else:
        break

cap.release()
# videoWriter.release()
cv2.destroyAllWindows()
