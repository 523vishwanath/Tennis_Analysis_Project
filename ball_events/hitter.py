from utils import get_center_of_bbox, measure_distance


def get_hitter(court_homography, frame_num, ball_detections, player_detections):
    # the player closest to the ball on the hit frame (ball_detections = INTERPOLATED)
    if 1 not in ball_detections[frame_num]:
        return None
    ball_center = get_center_of_bbox(ball_detections[frame_num][1])

    closest_track_id, closest_bbox, closest_distance = None, None, float('inf')
    for track_id, bbox in player_detections[frame_num].items():
        distance = measure_distance(ball_center, get_center_of_bbox(bbox))
        if distance < closest_distance:
            closest_track_id, closest_bbox, closest_distance = track_id, bbox, distance

    if closest_track_id is None:
        return None

    # hitter's feet on the court (feet are on the ground, so the homography is correct)
    x1, y1, x2, y2 = closest_bbox
    feet_x, feet_y = court_homography.image_to_court((x1 + x2) / 2, y2)
    side = 'far' if feet_y < court_homography.court_length / 2 else 'near'

    return {'player_id': closest_track_id, 'court_x': float(feet_x), 'court_y': float(feet_y), 'side': side}
