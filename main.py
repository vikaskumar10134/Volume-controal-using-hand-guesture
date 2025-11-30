import cv2
import mediapipe as mp
import time
import numpy as np
import hand_track_module as htm
import math
from pycaw.pycaw import AudioUtilities

#---------------------------------------
wCam , hCam = 640 , 720
#---------------------------------------

cap = cv2.VideoCapture(0)
previous_time = 0

cap.set(3 , wCam)
cap.set(4 , hCam)

# make object of hand detector class
detector = htm.HandDector(min_detection_confidence= 0.7)



device = AudioUtilities.GetSpeakers()
volume = device.EndpointVolume


# find out volume range
VolRange = volume.GetVolumeRange()
volBar = 400
vol = 0
volPer = 0
minVol = VolRange[0]
maxVol = VolRange[1]

# print('minvol' , minVol)
# print('maxvol' , maxVol)


while True:

    succes , frame = cap.read(0)

    # flip the image
    #frame = cv2.flip(frame , 1)

    # call the method to detect the hand
    frame = detector.findHands(frame)

    # call the method to detect the landmark of the hand
    landmark_list = detector.findPosition(frame , draw = False)

    if len(landmark_list) != 0:

        #print(landmark_list[4] , landmark_list[8])

        x1 , y1 = landmark_list[4][1] , landmark_list[4][2]
        x2 , y2 = landmark_list[8][1] , landmark_list[8][2]

        cx , cy = int((x1 + x2)/2) , int((y1 + y2)/2)

        cv2.circle(frame , (x1 , y1) , 10 , (255 , 0 , 255) , cv2.FILLED)   # draw circle on first position
        cv2.circle(frame , (x2 , y2) , 10 , (255 , 0 , 255) , cv2.FILLED)   # draw circle on second position
        cv2.line(frame , (x1 , y1) , (x2 , y2) , (255 , 0 , 255) , 3)   # plot line both position
        cv2.circle(frame , (cx , cy) , 10 , (255 , 0 , 255) , cv2.FILLED)   # draw circle on both center

        length = math.hypot(x2- x1 , y2 - y1)

        # print(length)

        # hand range (25 - 135)
        # volume range(-65 - 0)

        vol = np.interp(length , [25 , 135] , [minVol , maxVol])

        volBar = np.interp(length , [50 , 135] , [400 , 140])
        volPer = np.interp(length , [50 , 135] , [0 , 100])
        volume.SetMasterVolumeLevel(vol, None)
        #print(int(length) , vol)

        if length < 50:

            cv2.circle(frame , (cx , cy) , 10 , (0 , 255 , 0) , cv2.FILLED)

    cv2.rectangle(frame , (50 , 150) , (85 , 400) , (255 , 200 , 200) , 3)
    cv2.rectangle(frame , (50 , int(volBar)) , (85 , 400) , (255 , 200 , 200)  , cv2.FILLED)
    cv2.putText(frame , f'{int(volPer)}%' , (40 , 450) , cv2.FONT_HERSHEY_PLAIN , 1 , (255 , 255 ,100) , 2)


    current_time = time.time()

    fps = 1 / (current_time - previous_time)
    previous_time = current_time
    # show on frame 
    cv2.putText(frame , f'FPS : {int(fps)}' , (70 , 70) , cv2.FONT_HERSHEY_PLAIN , 2 , (255 , 255 ,0) , 2)


    # to show the frame
    cv2.imshow('frame ' , frame)

    # user press q it break the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):

        print('Quiting....')
        break

cap.release()
cv2.destroyAllWindows()