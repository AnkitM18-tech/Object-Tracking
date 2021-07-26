import cv2
from tracking import *

cap = cv2.VideoCapture("highway_bike.mp4")

#create tracker object
tracker = EuclideanDistTracker()

#object detection from stable camera
object_detector = cv2.createBackgroundSubtractorMOG2(history = 100,varThreshold=30)

while True:
    ret,frame = cap.read()

    #Extract region of interest
    # height,width, _ = frame.shape
    # print(height,width)

    roi = frame[340:720,500:800]

    #object detection
    mask = object_detector.apply(roi)
    _ , mask = cv2.threshold(mask,254,255,cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    detections = []

    for cnt in contours:
        #calculate area and remove small elements
        area = cv2.contourArea(cnt)
        if area > 100:
            #cv2.drawContours(roi,[cnt],-1,(255,0,0),2)
            x,y,w,h = cv2.boundingRect(cnt)
            detections.append([x,y,w,h])

    #object tracking
    boxes_id = tracker.update(detections)
    for box_id in boxes_id:
        x,y,w,h,id_ = box_id
        cv2.putText(roi,str(id_),(x,y-15),cv2.FONT_HERSHEY_PLAIN,1,(0,0,255),2)
        cv2.rectangle(roi,(x,y),(x+w,y+h),(255,0,0),2)

    cv2.imshow("Frame",frame)
    cv2.imshow("ROI",roi)

    key = cv2.waitKey(30)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()