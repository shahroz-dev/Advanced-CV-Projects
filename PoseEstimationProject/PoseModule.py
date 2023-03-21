import time
import cv2
import mediapipe as mp


class poseDetector:
    def __init__(self, mode=False, model_com=1, smooth_lm=True, en_seg=False,
                 smooth_seg=True, min_det_cf=0.5, min_tr_cf=0.5):
        self.results = None
        self.mode = mode
        self.model_com = model_com
        self.smooth_lm = smooth_lm
        self.en_seg = en_seg
        self.smooth_seg = smooth_seg
        self.min_det_cf = min_det_cf
        self.min_tr_cf = min_tr_cf
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(self.mode, self.model_com, self.smooth_lm, self.en_seg, self.smooth_seg,
                                     self.min_det_cf, self.min_tr_cf)
        self.mpDraw = mp.solutions.drawing_utils

    def findPose(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)
        return img

    def findPosition(self, img, draw=True):
        lm_list = []
        for id, lm in enumerate(self.results.pose_landmarks.landmark):
            h, w, c = img.shape
            # print(id, lm)
            # get the actual pixel value of landmarks instead of ratio
            cx, cy = int(lm.x * w), int(lm.y * h)
            lm_list.append([id, cx, cy])
            if draw:
                cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
        return lm_list


def main():
    frameWidth = 360
    cap = cv2.VideoCapture("pose_video.mp4")
    pTime = 0

    detector = poseDetector()

    while True:
        success, img = cap.read()
        img = detector.findPose(img)
        lm_list = detector.findPosition(img, draw=False)
        if len(lm_list) != 0:
            print(lm_list[14])
            cv2.circle(img, (lm_list[14][1], lm_list[14][2]), 15, (0, 0, 255), cv2.FILLED)

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


if __name__ == "__main__":
    main()
