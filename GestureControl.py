import cv2
import HandTrackingModule as htm

from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
import numpy as np

wCam, hCam = 640, 480

cap = cv2.VideoCapture(0)

cap.set(3, wCam)
cap.set(4, hCam)

detector = htm.HandDetector(detectionCon=0.7)
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_,CLSCTX_ALL,None)
volume = cast(interface, POINTER(IAudioEndpointVolume))

volRange = volume.GetVolumeRange()
minVol = volRange[0]
maxVol = volRange[1]
vol = 0
volBar = 400
volPer = 0

while True:
    success, img = cap.read()

    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)

    if lmList != ([],[]):
        length, img, [x1, y1, x2, y2, cx, cy]=detector.Distance(img,4,12,draw=True)

        vol = np.interp(length, [50, 300], [minVol, maxVol])
        volBar = np.interp(length, [50, 300], [400, 150])
        volPer = np.interp(length, [50, 300], [0, 150])
        print(int(length), vol)
        volume.SetMasterVolumeLevel(vol, None)


    cv2.imshow("Inset's Gesture Volume Control Program", img)

    k = cv2.waitKey(1)
    if k%256 == 27:
        print("Program shut down with no errors")
        break

cap.release()
cv2.destroyAllWindows()