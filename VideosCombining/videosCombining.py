import cv2
import numpy as np
import os

frameWidth = 320
frameHeight = 360

cap = cv2.VideoCapture(r'C:/Users/Muhammad Shahroz/Desktop/CV videos for profile/ai-fitness-trainer.mp4')
cap1 = cv2.VideoCapture(r'C:/Users/Muhammad Shahroz/Desktop/CV videos for profile/gestureVolControl.mp4')

fourcc = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')
dst_path = "Saved_Results/"
if not os.path.exists(dst_path):
    os.makedirs(dst_path)
videoWriter = cv2.VideoWriter(dst_path + 'CombineVideos.avi', fourcc, 30.0, (640, 360))
frame_counter = 0

while cap1.isOpened():

    frame_counter += 1
    # If the last frame is reached, reset the capture and the frame_counter
    if frame_counter == cap1.get(cv2.CAP_PROP_FRAME_COUNT):
        frame_counter = 0  # Or whatever as long as it is the same as next line
        cap1.set(cv2.CAP_PROP_POS_FRAMES, 0)

    ret, frame = cap.read()
    ret1, frame1 = cap1.read()
    if ret1:
        frame = cv2.resize(frame, (frameWidth, frameHeight), interpolation=cv2.INTER_AREA)
        frame1 = cv2.resize(frame1, (frameWidth, frameHeight), interpolation=cv2.INTER_AREA)
        both = np.concatenate((frame, frame1), axis=1)
        both = cv2.resize(both, (640, 360))

        cv2.imshow('Frame', both)
        videoWriter.write(both)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

cap.release()
videoWriter.release()
cv2.destroyAllWindows()
