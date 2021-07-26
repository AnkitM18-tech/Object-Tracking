import cv2

cap = cv2.VideoCapture("highway.mp4")

#object detection from stable camera
object_detector = cv2.createBackgroundSubtractorMOG2()

while True:
    ret,frame = cap.read()

    mask = object_detector.apply(frame)

    cv2.imshow("Frame",frame)

    key = cv2.waitKey(30)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()