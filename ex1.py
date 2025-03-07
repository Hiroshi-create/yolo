from ultralytics import YOLO
import cv2

model = YOLO("yolov8x-pose.pt")
results = model("https://ultralytics.com/images/ex1.jpg", save=True, save_txt=True, save_conf=True)
keypoints = results[0].keypoints

# print(keypoints.data[4][1])
left_shoulder = keypoints.data[0][0][0]
print(left_shoulder)

path = "ex1.jpg"
img = cv2.imread(path)


skeleton=[[[6,5],[6,8],[8,10],[5,7],[7,9],[6,12],[5,11],[12,11],[12,14],[14,16],[11,13],[13,15]]]
for number in range(0,12):
    s = skeleton[0][number][0]
    e = skeleton[0][number][1]
    cv2.line(
        img,
        (int(keypoints.data[0][s][0]), int(keypoints.data[0][s][1])),
        (int(keypoints.data[0][e][0]),int(keypoints.data[0][e][1])),
        (0, 0, 255),
        thickness = 4,
    )

for number in range(5,17):
    cv2.circle(img, (int(keypoints.data[0][number][0]), int(keypoints.data[0][number][1])), 4, (0, 0, 0), thickness = -1)


cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()