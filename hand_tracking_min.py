import cv2 as cv
import mediapipe as mp
import time

capture = cv.VideoCapture(0, cv.CAP_DSHOW)
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

preTime = 0
currTime = 0

while True:
    isTrue, frame = capture.read()
    frameRGB = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    results = hands.process(frameRGB)
    mpLandmarks = results.multi_hand_landmarks
    if mpLandmarks:
        for hand in mpLandmarks:
            mpDraw.draw_landmarks(frame, hand, mpHands.HAND_CONNECTIONS)
            for id, lm in enumerate(hand.landmark):
                height, width, channel = frame.shape
                cx, cy = int(lm.x * width), int(lm.y * height)
                # cv.circle(frame, (cx, cy), 15, (255,0,255), cv.FILLED)

    currTime = time.time()
    fps = 1/(currTime-preTime)
    preTime = currTime

    cv.putText(frame, str(int(fps)), (20,50),
               cv.FONT_HERSHEY_PLAIN, 3,
               (255,0,255),3)
    cv.imshow('Hand tracking', frame)
    if cv.waitKey(1) & 0xFF == ord('d'):
        break
capture.release()
cv.destroyAllWindows()