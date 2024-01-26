from ultralytics import YOLO
import cv2
import cvzone
import math
from sort import *

cap=cv2.VideoCapture('Videos/3.mp4')
# cap=cv2.VideoCapture(0)
cap.set(3,1280)
cap.set(4,720)

model=YOLO('Weights/yolov8l.pt')

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]


while True:
    _,img=cap.read()
    
    res=model(img,stream=True)
    
    
    for r in res:
        boxes=r.boxes
        for box in boxes:
            
            x1,y1,x2,y2=box.xyxy[0]
            
            #confidence
            conf=math.ceil(box.conf[0]*100)/100

            #class
            class_id=box.cls[0]

            
            x1,y1,x2,y2=int(x1),int(y1),int(x2),int(y2)
            # cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,255),3)
            
            #rectangle
            cvzone.cornerRect(img,(x1,y1,x2-x1,y2-y1),l=9,rt=5)
            
            # text
            cvzone.putTextRect(img,
                               f'{classNames[int(class_id)]}',(max(0,x1),max(35,y1)),
                               scale=1,
                               thickness=1,
                               offset=3)


    cv2.imshow("Image",img)
    # cv2.imshow("ImageRegion",img_region)
    key = cv2.waitKey(1)
    if key == 27:
        break
    elif key == 115:
        cv2.imwrite('Screenshots/screenshot.jpg', img) 

cap.release()
cv2.destroyAllWindows()