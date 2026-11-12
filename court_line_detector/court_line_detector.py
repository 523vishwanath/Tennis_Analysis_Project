import torch
import torchvision.transforms as transforms
import cv2
import sys
from torchvision import models
import numpy as np
sys.path.append("../")
import constants
from utils import convert_meters_to_pixel_distance, measure_distance


class CourtLineDetector:
    def __init__(self, model_path):
        self.model = models.resnet50(pretrained=True)
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, 14*2) 
        self.model.load_state_dict(torch.load(model_path, map_location='cpu'))
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def predict(self, image):

    
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_tensor = self.transform(image_rgb).unsqueeze(0)
        with torch.no_grad():
            outputs = self.model(image_tensor)
        keypoints = outputs.squeeze().cpu().numpy()
        original_h, original_w = image.shape[:2]
        keypoints[::2] *= original_w / 224.0
        keypoints[1::2] *= original_h / 224.0

        return keypoints

    def draw_keypoints(self, image, keypoints):
        # Plot keypoints on the image
        for i in range(0, len(keypoints), 2):
            x = int(keypoints[i])
            y = int(keypoints[i+1])
            cv2.putText(image, str(i//2), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            #court line
            cv2.line(image, (750,450),(1100,450), (0, 255, 0), 2)
            cv2.circle(image, (x, y), 5, (0, 0, 255), -1)
            if i==24:
                p= int(keypoints[i+2])
                q= int(keypoints[i+3])
                Net_center_x, Net_center_y = int((x + p) / 2), int((y + q) / 2)
                pixel_distance_between_12_and_13 = int(measure_distance((x, y), (p, q)))
                pixel_net_height = convert_meters_to_pixel_distance(constants.COURT_NET_HEIGHT, constants.DISTANCE_BETWEEN_POINT_12_AND_13,
                                                                    pixel_distance_between_12_and_13)
                net_top_x = Net_center_x
                net_top_y = int(Net_center_y - pixel_net_height)
                cv2.circle(image, (net_top_x, net_top_y), 5, (255, 0, 0), -1)
        return image
    
    def draw_keypoints_on_video(self, video_frames, keypoints):
        output_video_frames = []
        for frame in video_frames:
            frame = self.draw_keypoints(frame, keypoints)
            output_video_frames.append(frame)
        return output_video_frames