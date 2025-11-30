import cv2
import mediapipe as mp
import time


class HandDector():

    # constructor
    def __init__(self , static_image_mode = False , 
                max_num_hands = 2 , 
                min_detection_confidence = 0.5 , 
                min_tracking_confidence = 0.5):

        self.static_image_mode = static_image_mode
        self.max_num_hands = max_num_hands
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        

        self.mp_hands = mp.solutions.hands

        self.hands = self.mp_hands.Hands(static_image_mode = self.static_image_mode , 
                                         max_num_hands = self.max_num_hands , 
                                         min_detection_confidence = self.min_detection_confidence , 
                                         min_tracking_confidence = self.min_tracking_confidence)


        self.mp_draw = mp.solutions.drawing_utils

    def findHands(self , frame , draw = True):

        # convert into rgb
        frame_rgb = cv2.cvtColor(frame , cv2.COLOR_BGR2RGB)

        self.result = self.hands.process(frame_rgb)
        
        # for check the hand
        if self.result.multi_hand_landmarks:

            for handLandMarks in self.result.multi_hand_landmarks:

                if draw:
                    # draw the landmark
                    self.mp_draw.draw_landmarks(frame , handLandMarks , self.mp_hands.HAND_CONNECTIONS)

        return frame
    

    def findPosition(self , frame , Handno = 0 , draw = True):

        landmark_list = []

        if self.result.multi_hand_landmarks:

            myHands = self.result.multi_hand_landmarks[Handno]
  
            # get the id and landmark info
            for id , landmark in enumerate(myHands.landmark):

                height , width , channel = frame.shape

                # find out the position in px
                cx , cy = int(landmark.x* width) , int(landmark.y * height)

                landmark_list.append([id , cx , cy])

                if draw:

                    cv2.circle(frame , (cx , cy) , 7 , (255 , 0 , 0) , -1)

        return landmark_list




def main():

    previous_time = 0
    current_time = 0

    video = cv2.VideoCapture(0)

    detector = HandDector()

    while True:

        succes , frame = video.read()

        if not succes:
            print('Warning: cant read frame (camera disconnected?)')
            break

        frame = detector.findHands(frame)

        landmark_list = detector.findPosition(frame)

        if len(landmark_list) != 0:
            print(landmark_list[4])

        # calculate the fps
        current_time = time.time()
        fps = 1 / (current_time - previous_time)
        previous_time = current_time

        cv2.putText(frame , str(int(fps)) , (10 , 70) , cv2.FONT_HERSHEY_COMPLEX , 3 , (50 , 255 , 0) , 3)

        # to show the frame
        cv2.imshow('frame ' , frame)

        # press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()



if __name__ == '__main__':

    main()


