from ultralytics import YOLO
import cv2

# YOLOモデルの初期化
model = YOLO("yolov8x-pose.pt")

# 元の画像を取得し、キーポイントを抽出
jpgResults = model("https://ultralytics.com/images/ex1.jpg", save=True, save_txt=True, save_conf=True)
jpgKeypoints = jpgResults[0].keypoints
path = "ex1.jpg"
originalImg = cv2.imread(path)
skeleton = [[[6, 5], [6, 8], [8, 10], [5, 7], [7, 9], [6, 12], [5, 11], [12, 11], [12, 14], [14, 16], [11, 13], [13, 15]]]

# # 座標4点の真ん中の座標
# def calculate_center(p1, p2, p3, p4):
#     x_coords = [p1[0], p2[0], p3[0], p4[0]]
#     y_coords = [p1[1], p2[1], p3[1], p4[1]]
    
#     center_x = sum(x_coords) / len(x_coords)
#     center_y = sum(y_coords) / len(y_coords)
#     return (center_x, center_y)


def calculate_center(p1, p2, p3, p4):
    # 対角線の交点を計算する
    x = ((p1[0] + p3[0]) / 2 + (p2[0] + p4[0]) / 2) / 2
    y = ((p1[1] + p3[1]) / 2 + (p2[1] + p4[1]) / 2) / 2

    return (x, y)


# スケルトンを描画する関数
def draw_skeleton(keypoints, img):
    originalMidpoint = calculate_center((int(jpgKeypoints.data[0][5][0]), int(jpgKeypoints.data[0][5][1])), (int(jpgKeypoints.data[0][6][0]), int(jpgKeypoints.data[0][6][1])), (int(jpgKeypoints.data[0][11][0]), int(jpgKeypoints.data[0][11][1])), (int(jpgKeypoints.data[0][12][0]), int(jpgKeypoints.data[0][12][1])))

    midpoint = calculate_center((int(keypoints.data[0][5][0]), int(keypoints.data[0][5][1])), (int(keypoints.data[0][6][0]), int(keypoints.data[0][6][1])), (int(keypoints.data[0][11][0]), int(keypoints.data[0][11][1])), (int(keypoints.data[0][12][0]), int(keypoints.data[0][12][1])))

    # 差分
    diff_x = midpoint[0] - originalMidpoint[0] 
    diff_y = midpoint[1] - originalMidpoint[1]


    # 色を変えるため
    sum = 0
    for number in range(5, 17):
        sum += abs(int(keypoints.data[0][number][0] + diff_x) - int(jpgKeypoints.data[0][number][0] + diff_x)) + abs(int(keypoints.data[0][number][1] + diff_y) - int(jpgKeypoints.data[0][number][1] + diff_y))

    color = (0, 0, 255) if (sum >= 150) else (0, 255, 0)

    # 線を描画
    for number in range(0, 12):
        s = skeleton[0][number][0]
        e = skeleton[0][number][1]
        cv2.line(
            img,
            (int(keypoints.data[0][s][0] + diff_x), int(keypoints.data[0][s][1] + diff_y)),
            (int(keypoints.data[0][e][0] + diff_x), int(keypoints.data[0][e][1] + diff_y)),
            color,
            thickness=4,
        )
        cv2.line(
            img,
            (int(jpgKeypoints.data[0][s][0] + diff_x), int(jpgKeypoints.data[0][s][1] + diff_y)),
            (int(jpgKeypoints.data[0][e][0] + diff_x), int(jpgKeypoints.data[0][e][1] + diff_y)),
            (255, 0, 0),
            thickness=4,
        )
    # 関節に印
    for number in range(5, 17):
        cv2.circle(img, (int(keypoints.data[0][number][0] + diff_x), int(keypoints.data[0][number][1] + diff_y)), 4, (0, 0, 0), thickness=-1)
        cv2.circle(img, (int(jpgKeypoints.data[0][number][0] + diff_x), int(jpgKeypoints.data[0][number][1] + diff_y)), 4, (255, 0, 0), thickness=-1)

# 動画ファイルを読み込む
video_path = 'ex3a.mp4'
cap = cv2.VideoCapture(video_path)
frames = []

# 動画からフレームを読み込む
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frames.append(frame)
cap.release()

# フレームごとに処理
for frameIndex in range(len(frames)):
    # モデルを使用してキーポイントを取得
    results = model(frames[frameIndex], save=True, save_txt=True, save_conf=True)
    keypoints = results[0].keypoints
    # draw_skeleton(jpgKeypoints, img)
    # スケルトンを描画
    draw_skeleton(keypoints, frames[frameIndex])

    # 描画したフレームを表示
    cv2.imshow('image', frames[frameIndex])
    if 0xFF == ord('q'):  # 'q'キーで終了
        break
    cv2.waitKey(1)
cv2.destroyAllWindows()
