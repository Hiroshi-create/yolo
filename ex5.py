from ultralytics import YOLO
import cv2


# 動画ファイルを読み込む
video_path = 'ex5.mp4'
cap = cv2.VideoCapture(video_path)
frames = []

# 動画からフレームを読み込む
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frames.append(frame)
cap.release()


model = YOLO("yolov8x.pt")

for frameIndex in range(len(frames)):
    results = model(frames[frameIndex])

    for box in results[0].boxes:
        if box.cls[0] == 0:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frames[frameIndex], (x1, y1), (x2, y2), (0, 0, 255), 4)

    cv2.imshow('image', frames[frameIndex])
    if 0xFF == ord('q'):  # 'q'キーで終了
        break
    cv2.waitKey(1)
cv2.destroyAllWindows()