import cv2 as cv
import mediapipe as mp
import time

class handDetector():
    def __init__(self, mode=False, maxHands=2, detectionConf=0.5, trackingConf=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionConf = detectionConf
        self.trackingConf = trackingConf

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands()
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self,frame, draw=True):
        frameRGB = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        self.results = self.hands.process(frameRGB)
        self.mpLandmarks = self.results.multi_hand_landmarks
        if self.mpLandmarks:
            for hand in self.mpLandmarks:
                if draw == True:
                    self.mpDraw.draw_landmarks(frame, hand,self.mpHands.HAND_CONNECTIONS)
        return frame

    def findPosition(self, frame, handNo=0, draw=True):
        lmList = []
        if self.mpLandmarks:
            myHand = self.mpLandmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                height, width, channel = frame.shape
                cx, cy = int(lm.x * width), int(lm.y * height)
                # print(id, cx, cy)
                lmList.append([id, cx, cy])
                if draw == True:
                    cv.circle(frame, (cx, cy), 5, (255,0,255), cv.FILLED)
        return lmList



def main():
    capture = cv.VideoCapture(0, cv.CAP_DSHOW)
    preTime = 0
    currTime = 0
    detector = handDetector()
    while True:
        isTrue, frame = capture.read()
        frame = detector.findHands(frame)
        lmList = detector.findPosition(frame)
        if len(lmList) != 0:
            print(lmList[4])
        currTime = time.time()
        fps = 1 / (currTime - preTime)
        preTime = currTime

        cv.putText(frame, str(int(fps)), (20, 50),
                   cv.FONT_HERSHEY_PLAIN, 3,
                   (255, 0, 255), 3)
        cv.imshow('Hand tracking', frame)
        if cv.waitKey(1) & 0xFF == ord('d'):
            break
    capture.release()
    cv.destroyAllWindows()

if __name__ == '__main__':
    main()