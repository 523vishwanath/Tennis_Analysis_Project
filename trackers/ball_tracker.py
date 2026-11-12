from ultralytics import YOLO
import cv2
import pickle
import pandas as pd
from scipy.signal import savgol_filter
import numpy as np
from utils import get_device

class BallTracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.device = get_device()          # 'mps' = Apple GPU, much faster and cooler than the CPU


    def interpolate_ball_positions(self, ball_positions):
        ball_positions = [x.get(1,[np.nan]*4) for x in ball_positions]
        # convert the list into pandas dataframe
        df_ball_positions = pd.DataFrame(ball_positions,columns=['x1','y1','x2','y2'])

        # interpolate the missing values
        df_ball_positions = df_ball_positions.interpolate()
        df_ball_positions = df_ball_positions.bfill()

        ball_positions = [{1:x} for x in df_ball_positions.to_numpy().tolist()]

        return ball_positions


    def get_ball_shot_frames(self, df_ball_positions):

        ball_positions = [x.get(1,[np.nan]*4) for x in df_ball_positions]
        # convert the list into pandas dataframe
        df_ball_positions = pd.DataFrame(ball_positions,columns=['x1','y1','x2','y2'])

        # interpolate the missing values
        df_ball_positions = df_ball_positions.interpolate()
        df_ball_positions = df_ball_positions.bfill()

        df_ball_positions['mid_y'] = (df_ball_positions['y1'] + df_ball_positions['y2']) / 2
        df_ball_positions['mid_y_rolling_mean'] = df_ball_positions['mid_y'].rolling(window =5,min_periods=1, center =False).mean()

        df_ball_positions['delta_y'] = df_ball_positions['mid_y_rolling_mean'].diff()

        #Logic to find the ball shot frames
        df_ball_positions['ball_hit'] = 0
        df_ball_positions['ball_hit_other_side'] = 0
        df_ball_positions['ball_hit_this_side'] = 0


        minimum_change_frames_for_hit = 25
        window = int(minimum_change_frames_for_hit * 1.2)

        delta_y = df_ball_positions['delta_y']

        for i in range(1, len(df_ball_positions) - window - 1):

            # Detect a direction reversal
            negative_change = delta_y.iloc[i] > 0 and delta_y.iloc[i + 1] < 0
            positive_change = delta_y.iloc[i] < 0 and delta_y.iloc[i + 1] > 0

            if negative_change or positive_change:
                change_count = 0
                # Count how many subsequent frames continue in the new direction
                following= delta_y.iloc[i + 2:i + window + 1]

                if negative_change:
                    change_count = (following < 0).sum() 
                else:
                    change_count = (following > 0).sum()

                if change_count >= minimum_change_frames_for_hit and positive_change:
                    df_ball_positions.loc[i, 'ball_hit'] = 1
                    df_ball_positions.loc[i, 'ball_hit_other_side'] = 1

                if change_count >= minimum_change_frames_for_hit and negative_change:
                    df_ball_positions.loc[i, 'ball_hit'] = 1
                    df_ball_positions.loc[i, 'ball_hit_this_side'] = 1

        frame_nums_with_ball_hits = df_ball_positions[df_ball_positions['ball_hit'] == 1].index.tolist()
        frame_nums_with_ball_hits_this_side = df_ball_positions[df_ball_positions['ball_hit_this_side'] == 1].index.tolist()
        frame_nums_with_ball_hits_other_side = df_ball_positions[df_ball_positions['ball_hit_other_side'] == 1].index.tolist()

        return frame_nums_with_ball_hits, frame_nums_with_ball_hits_this_side, frame_nums_with_ball_hits_other_side


    def get_ball_bounce_frames(
    self,
    ball_positions,
    shot_frames,
    min_gap_from_shot=8,
    min_velocity=1.5,
    min_acceleration=2.0,
    min_bounce_distance=10
):

        positions = [x.get(1, []) for x in ball_positions]

        df = pd.DataFrame(
            positions,
            columns=['x1', 'y1', 'x2', 'y2']
        )

        # Interpolate missing detections
        df = df.interpolate(limit_direction='both')

        # Calculate ball center
        y = ((df['y1'] + df['y2']) / 2).to_numpy()

        if len(y) < 7:
            return []

        # Smooth trajectory
        y_smooth = savgol_filter(y, window_length=7, polyorder=2)

        # Calculate velocity and acceleration
        velocity = np.gradient(y_smooth)
        acceleration = np.gradient(velocity)

        # Bounce: downward movement followed by upward movement
        bounce_candidates = []

        for i in range(2, len(y_smooth) - 2):

            # Ball must approach the ground (positive image velocity)
            moving_down = velocity[i - 1] > min_velocity

            # Ball must reverse direction after impact
            moving_up = velocity[i + 1] < -min_velocity

            # Sudden acceleration around the impact
            impact = abs(acceleration[i]) > min_acceleration

            if moving_down and moving_up and impact:
                bounce_candidates.append(i)

        # Divide trajectory using shot frames
        shots = sorted(set(shot_frames))
        boundaries = shots + [len(y_smooth) - 1]

        bounce_frames = []

        for start, end in zip(boundaries[:-1], boundaries[1:]):

            lo = start + min_gap_from_shot
            hi = end - min_gap_from_shot

            candidates = [
                frame for frame in bounce_candidates
                if lo <= frame <= hi
            ]

            for candidate in candidates:

                if not bounce_frames or (
                    candidate - bounce_frames[-1] >= min_bounce_distance
                ):
                    bounce_frames.append(candidate)

        return bounce_frames

    def detect_frames(self, frames, read_from_stub = False, stub_path = None):
            ball_detections = []


            if read_from_stub and stub_path is not None:
                with open(stub_path, 'rb') as f:
                    BallTracker_detections = pickle.load(f)
                return BallTracker_detections


            
            for frame in frames:
                ball_dict = self.detect_frame(frame)
                ball_detections.append(ball_dict)


            if stub_path is not None:
                with open(stub_path, 'wb') as f:
                    pickle.dump(ball_detections, f)

            return ball_detections



    def detect_frame(self,frame):
        results = self.model.predict(frame, conf = 0.15, device = self.device, verbose = False)[0]

        ball_dict = {}
        for box in results.boxes:
            
            result = box.xyxy.tolist()[0]
            ball_dict[1] = result

        return ball_dict

    def detect_frames_candidates(self, frames, read_from_stub = False, stub_path = None):
        # like detect_frames, but keeps EVERY ball per frame (needed to ignore balls lying on the court)
        if read_from_stub and stub_path is not None:
            with open(stub_path, 'rb') as f:
                return pickle.load(f)

        ball_candidates = [self.detect_frame_candidates(frame) for frame in frames]

        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(ball_candidates, f)
        return ball_candidates

    def detect_frame_candidates(self, frame):
        # EVERY ball the model sees in this frame: [x1, y1, x2, y2, confidence]
        # (detect_frame keeps only one box, which can be a ball lying on the court)
        results = self.model.predict(frame, conf = 0.15, device = self.device, verbose = False)[0]

        candidates = []
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy.tolist()[0]
            confidence = float(box.conf.tolist()[0])
            candidates.append([x1, y1, x2, y2, confidence])

        return candidates

    def draw_bboxes(self, frames, ball_detections):
        output_video_frames = []
        for frame, ball_dict in zip(frames, ball_detections):
        #draw bounding boxes
            for track_id,bbox in ball_dict.items():
                x1, y1, x2, y2 = bbox
                frame = cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)),(0, 255, 255), 2)
                frame = cv2.putText(frame, f"Ball {track_id}", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            output_video_frames.append(frame)

        return output_video_frames
    


    
