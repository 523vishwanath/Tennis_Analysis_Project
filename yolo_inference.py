from ultralytics import YOLO
import torch

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Running YOLO on: {device}")

model = YOLO("yolo26l.pt")
result = model.track(source = "input_videos/input_video.mp4",conf =0.2, save = True) 

#print(result)
#print("----------------------------boxes:---------------------------")
#for box in result[0].boxes:
#    print(box)