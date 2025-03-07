from ultralytics import YOLO
import cv2

model = YOLO("yolov8x.pt")
results = model("ex4.jpg")
img = cv2.imread("ex4.jpg")

for box in results[0].boxes:
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 4)

cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()