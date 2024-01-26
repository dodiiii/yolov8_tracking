from ultralytics import YOLO
import cv2
import cvzone
import math
from sort import *

cap=cv2.VideoCapture('Videos/2.mp4')
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

indices_of_interest=[2,3,5,7]

mask = cv2.imread('mask.png')

tracker=Sort(max_age=20, min_hits=3, iou_threshold=0.3)

limits=[255,250,450,250]

# total_count={2:[],3:[],5:[],7:[]}
total_count=[]

while True:
    _,img=cap.read()
    img_region=cv2.bitwise_and(img,mask)
    
    res=model(img_region,stream=True)
    
    detections=np.empty((0,5))
    
    for r in res:
        boxes=r.boxes
        for box in boxes:
            
            x1,y1,x2,y2=box.xyxy[0]
            
            #confidence
            conf=math.ceil(box.conf[0]*100)/100

            #class
            class_id=box.cls[0]
            
            #detect only traffic classes
            if int(class_id) not in indices_of_interest or conf<0.25:
                continue
            
            x1,y1,x2,y2=int(x1),int(y1),int(x2),int(y2)
            # cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,255),3)
            
            #rectangle
            # cvzone.cornerRect(img,(x1,y1,x2-x1,y2-y1),l=9,rt=5)
            
            # text
            cvzone.putTextRect(img,
                               f'{classNames[int(class_id)]}',(max(0,x1),max(35,y1)),
                               scale=1,
                               thickness=1,
                               offset=3)
            
            current_arr=np.array([x1,y1,x2,y2,conf])
            detections=np.vstack((detections,current_arr))

            
    tracker_res=tracker.update(detections)
    
    cv2.line(img,(limits[0],limits[1]),(limits[2],limits[3]),
             (0,0,255),5)
    
    for res in tracker_res:
        x1,y1,x2,y2,id=res
        x1,y1,x2,y2=int(x1),int(y1),int(x2),int(y2)
        cvzone.cornerRect(img,(x1,y1,x2-x1,y2-y1),l=9,rt=2,colorR=(255,0,255))
        cvzone.putTextRect(img,
                    f'id: {int(id)}',(max(0,x1),max(35,y1-20)),
                    scale=1,
                    thickness=1,
                    offset=3)
        
        cx,cy=x1+(x2-x1)//2,y1+(y2-y1)//2
        cv2.circle(img,(cx,cy),5,(255,0,255),cv2.FILLED)
        
        if limits[0]<cx<limits[2] and limits[1]-10<cy<limits[3]+10:
            if int(id) not in total_count:
                total_count.append(int(id))
                cv2.line(img,(limits[0],limits[1]),(limits[2],limits[3]),
             (0,255,0),5)
                
    # cvzone.putTextRect(img,
    # f"Cars: {len(total_count[2])}", (50, 50),scale=2)
    
    # cvzone.putTextRect(img,
    # f"Motorbikes: {len(total_count[3])}", (50, 100),scale=2)
    
    # cvzone.putTextRect(img,
    # f"Bus: {len(total_count[5])}", (50, 150),scale=2)
    
    # cvzone.putTextRect(img,
    # f"Trucks: {len(total_count[7])}", (50, 200),scale=2)
    cvzone.putTextRect(img,
    f"Total count: {len(total_count)}", (50, 50),scale=2)
     
    cv2.imshow("Image",img)
    # cv2.imshow("ImageRegion",img_region)
    key = cv2.waitKey(1)
    if key == 27:
        break
    elif key == 115:
        cv2.imwrite('Screenshots/screenshot.jpg', img) 

cap.release()
cv2.destroyAllWindows()