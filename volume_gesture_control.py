import cv2 as cv
import numpy as np
import time
import math
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
from hand_tracking_module import handDetector

camWidth, camHeight = 640, 480

capture = cv.VideoCapture(0, cv.CAP_DSHOW)
capture.set(3, camWidth)
capture.set(4, camHeight)


devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))

volRange = volume.GetVolumeRange()
minVol = volRange[0]
maxVol = volRange[1]
vol = 0
volBar = 400
volPercent = 0

currTime = 0
prevTime = 0

detector = handDetector(detectionConf=0.7)

while True:
    isTrue, frame = capture.read()
    frame = detector.findHands(frame)
    lmList = detector.findPosition(frame, draw=False)
    if len(lmList) != 0:

        x1, y1 = lmList[4][1], lmList[4][2]
        x2, y2 = lmList[8][1], lmList[8][2]
        cx, cy = (x1+x2) // 2, (y1+y2) // 2

        cv.circle(frame, (x1, y1), 10, (255, 0, 255), cv.FILLED)
        cv.circle(frame, (x2, y2), 10, (255, 0, 255), cv.FILLED)
        cv.line(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)
        cv.circle(frame, (cx, cy), 10, (255, 0, 200), cv.FILLED)

        length = math.hypot(x2-x1, y2-y1)

        # min and max lengths 10 - 180
        vol = np.interp(length, [7, 180], [minVol, maxVol])
        volBar = np.interp(length, [7, 180], [400, 150])
        volPercent = np.interp(length, [7, 180], [0, 100])
        print(length, vol)

        volume.SetMasterVolumeLevel(vol, None)

        if length < 20:
            cv.circle(frame, (cx, cy), 10, (0, 255, 0), cv.FILLED)

    cv.rectangle(frame, (20, 150), (50, 400), (0,0,0), 3)
    cv.rectangle(frame, (20, int(volBar)), (50, 400), (0,255,30), cv.FILLED)
    cv.putText(frame, f'Vol : {str(int(volPercent))}', (10, 120),
               fontFace=cv.FONT_HERSHEY_COMPLEX, fontScale=1,
               thickness=2, color=(0,0,0))

    currTime = time.time()
    fps = str(int(1/(currTime - prevTime)))
    prevTime = currTime

    cv.putText(frame, f'FPS:{fps}', (10, 50), color=(255, 0, 255),
               fontFace=cv.FONT_HERSHEY_COMPLEX,
               fontScale=1,
               thickness=2)
    cv.imshow('Volume Gesture Control', frame)
    if cv.waitKey(1) & 0xFF == ord('d'):
        break

capture.release()
cv.destroyAllWindows()
