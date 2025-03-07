from ultralytics import YOLO
import cv2
import torch

model = YOLO("yolov8x-pose.pt")

mainResults = model("https://ultralytics.com/images/ex1.jpg", save=True, save_txt=True, save_conf=True)
keypoints = mainResults[0].keypoints
data = keypoints.data

jpgNameList = ["ex2_307.jpg", "ex2_336.jpg", "ex2_2015.jpg", "ex2_3077.jpg", "ex2_5175.jpg"]

for jpgIndex in range(0, len(jpgNameList)):
    results = model("https://ultralytics.com/images/" + jpgNameList[jpgIndex], save=True, save_txt=True, save_conf=True)
    keypoints = results[0].keypoints
    data = torch.cat((data, keypoints.data), dim=0)

absKeypointsList = []
for jpgIndex in range(1, len(data)):
    sum = 0
    for keypointsNumber in range(5,17):
        sum += abs(data[0][keypointsNumber][0] - data[jpgIndex][keypointsNumber][0]) + abs(data[0][keypointsNumber][1] - data[jpgIndex][keypointsNumber][1])
    absKeypointsList.append(sum)

sorted_list = sorted(absKeypointsList, key=abs)
print(sorted_list)
index_mapping = {value: i for i, value in enumerate(absKeypointsList)}
replaced_list = [index_mapping[value] for value in sorted_list]

for replaced in replaced_list:
    print(jpgNameList[replaced]+"\n")