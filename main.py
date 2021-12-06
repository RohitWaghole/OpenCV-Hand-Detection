import cv2
import mediapipe as mp
import HandDetectionModule as HandDetect

cap = cv2.VideoCapture(0)
detection = HandDetect.HandDetection()

while True:
        _, img = cap.read()

        img = cv2.flip(img, 1)

        img = detection.HandsFinding(img)
        landmarkList = detection.FindPosition(img)

        # here landmarkList contains 3 parameters for each landmark
        # landmarkList[landmarknumber, x-axis pos, y-axis pos]
        # we have 0 to 20 i.e total 21 landmarks on hand
        # landmark number 4 is for the thumb tip so it will draw a circle on the tip of the thumb
        if len(landmarkList):
            cv2.circle(img, (landmarkList[4][1], landmarkList[4][2]), 7, (255, 243, 15), -1)

        cv2.imshow('Capturing', img)
        k = cv2.waitKey(30)
        if k == 13:
            break

cap.release()
cv2.destroyAllWindows()
