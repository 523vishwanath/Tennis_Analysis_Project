from ultralytics import YOLO
import cv2
import numpy as np
import pickle
import sys
sys.path.append("../")
from utils import get_center_of_bbox, measure_distance, get_device


class PlayerTracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.device = get_device()          # 'mps' = Apple GPU, much faster and cooler than the CPU


    def choose_and_filter_players(self, player_detections, court_keypoints):
        player_detections_first_frame = player_detections[0]
        choosen_player = self.choose_player(player_detections_first_frame, court_keypoints)

        filtered_player_detections = []
        for player_dict in player_detections:
            filtered_player_dict = {track_id: bbox for track_id, bbox in player_dict.items()
                                     if track_id in choosen_player}
            filtered_player_detections.append(filtered_player_dict)

        return filtered_player_detections

    def choose_players_by_court(self, player_detections, court_homography):
        """
        Pick the 2 players by WHERE THEY STAND, not by tracker id (ids switch, and line judges / ball kids get ids too).
        Returns per frame {1: near player bbox, 2: far player bbox}, so P1 is always the near player and P2 the far one.
        """
        court_width = court_homography.court_width
        court_length = court_homography.court_length

        sides = [
            (1, 'near', court_length / 2, court_length + 6),   # player 1: near half, up to 6 m behind the baseline
            (2, 'far', -6, court_length / 2),                  # player 2: far half, up to 6 m behind the baseline
        ]
        max_jump_meters = 4.0        # a player can't move more than this between 2 detections
        forget_after_frames = 30     # not seen for this long -> pick again from scratch

        previous_position = {1: None, 2: None}
        frames_missing = {1: 0, 2: 0}
        chosen_player_detections = []

        for player_dict in player_detections:

            # every detected person's feet on the court, in meters
            people = []
            for track_id, bbox in player_dict.items():
                x1, y1, x2, y2 = bbox
                feet_x, feet_y = court_homography.image_to_court((x1 + x2) / 2, y2)
                people.append((bbox, float(feet_x), float(feet_y)))

            chosen = {}
            for player_id, side, y_min, y_max in sides:
                best = None
                for bbox, feet_x, feet_y in people:
                    if not (y_min <= feet_y <= y_max and -3 <= feet_x <= court_width + 3):
                        continue
                    if previous_position[player_id] is not None:
                        # follow the same person: closest to where the player was
                        score = np.hypot(feet_x - previous_position[player_id][0], feet_y - previous_position[player_id][1])
                        if score > max_jump_meters:
                            continue
                    else:
                        # first pick: must stand between the sidelines, closest to the middle
                        if not (0 <= feet_x <= court_width):
                            continue
                        score = abs(feet_x - court_width / 2)
                    if best is None or score < best[0]:
                        best = (score, bbox, (feet_x, feet_y))

                if best is not None:
                    chosen[player_id] = best[1]
                    previous_position[player_id] = best[2]
                    frames_missing[player_id] = 0
                else:
                    frames_missing[player_id] += 1
                    if frames_missing[player_id] > forget_after_frames:
                        previous_position[player_id] = None

            chosen_player_detections.append(chosen)

        return chosen_player_detections

    def choose_player(self, player_dict,court_keypoints):
        distances = []
        for track_id, bbox in player_dict.items():
            player_center = get_center_of_bbox(bbox)
            min_distance = float('inf')

            for i in range(0, len(court_keypoints), 2):
                court_keypoint = court_keypoints[i], court_keypoints[i+1]
                distance = measure_distance(player_center, court_keypoint)
                if distance < min_distance:
                    min_distance = distance

            distances.append((track_id,min_distance))
        # Sort the distances based on the distance value in ascending order
        distances.sort(key=lambda x: x[1])
        # Choose the first 2 tracks( 2 track ids with minimum distance from the court)
        chosen_players = [distances[0][0], distances[1][0]]
        return chosen_players

    def detect_frames(self, frames, read_from_stub = False, stub_path = None):
        player_detections = []


        if read_from_stub and stub_path is not None:
            with open(stub_path, 'rb') as f:
                player_detections = pickle.load(f)
            return player_detections


        
        for frame in frames:
            player_dict = self.detect_frame(frame)
            player_detections.append(player_dict)


        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(player_detections, f)    

        return player_detections



    def detect_frame(self,frame):
        results = self.model.track(frame, persist =True, device = self.device, verbose = False)[0]
        id_name_dict = results.names

        player_dict = {}
        for box in results.boxes:
            if box.id is None:          # tracker hasn't given this box an id yet
                continue
            track_id = int(box.id.tolist()[0])
            result = box.xyxy.tolist()[0]
            object_cls_id = box.cls.tolist()[0]
            object_cls_name = id_name_dict[object_cls_id]
            if object_cls_name == "person":
                player_dict[track_id] = result

        return player_dict

    def draw_bboxes(self, frames, player_detections):
        output_video_frames = []
        for frame, player_dict in zip(frames, player_detections):
        #draw bounding boxes
            for track_id,bbox in player_dict.items():
                x1, y1, x2, y2 = bbox
                frame = cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)),(0, 0, 255), 2)
                frame = cv2.putText(frame, f"Player {track_id}", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                
            output_video_frames.append(frame)

        return output_video_frames