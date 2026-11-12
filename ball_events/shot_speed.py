import numpy as np
from .hitter import get_hitter


class ShotSpeedCalculator:
    """Average speed of each shot = distance from the hitter's feet to the bounce / time from hit to bounce."""

    def __init__(self, court_homography, fps):
        self.court = court_homography
        self.fps = fps

        # sanity limits, anything outside is thrown away
        self.min_flight_seconds = 0.2
        self.max_flight_seconds = 3.0
        self.min_speed_kmh = 20
        self.max_speed_kmh = 260

    def get_hitter(self, frame_num, ball_detections, player_detections):
        return get_hitter(self.court, frame_num, ball_detections, player_detections)

    def get_shot_speeds(self, ball_shot_frames, ball_detections, player_detections, ball_bounces):
        # ball_detections = INTERPOLATED (needs a ball position on the hit frame)
        shot_frames = sorted(ball_shot_frames)
        shots = []
        previous_bounce_frame = -1

        for bounce in ball_bounces:
            # the hit that sent the ball to this bounce = last hit after the previous bounce
            hits_before = [f for f in shot_frames if previous_bounce_frame < f < bounce['frame']]
            previous_bounce_frame = bounce['frame']

            if not hits_before:
                continue        # e.g. the first bounce of the video, its hit happened before the clip

            hit_frame = hits_before[-1]
            hitter = self.get_hitter(hit_frame, ball_detections, player_detections)
            if hitter is None:
                continue

            # the ball has to land on the OTHER side of the net from the hitter
            if hitter['side'] == bounce['side']:
                continue        # wrong hit frame (missed hit / volley), skip instead of a fake speed

            flight_seconds = (bounce['frame'] - hit_frame) / self.fps
            if not (self.min_flight_seconds <= flight_seconds <= self.max_flight_seconds):
                continue

            distance_meters = float(np.hypot(bounce['court_x'] - hitter['court_x'], bounce['court_y'] - hitter['court_y']))
            speed_kmh = distance_meters / flight_seconds * 3.6

            if not (self.min_speed_kmh <= speed_kmh <= self.max_speed_kmh):
                continue

            shots.append({
                'hit_frame': hit_frame,
                'bounce_frame': bounce['frame'],
                'player_id': hitter['player_id'],
                'player_side': hitter['side'],
                'distance_meters': distance_meters,
                'flight_seconds': flight_seconds,
                'speed_kmh': speed_kmh,
            })

        return shots

    @staticmethod
    def get_average_speeds(shots):
        # {player_id: (average km/h, number of shots, side)}
        averages = {}
        for player_id in sorted(set(s['player_id'] for s in shots)):
            player_shots = [s for s in shots if s['player_id'] == player_id]
            averages[player_id] = (float(np.mean([s['speed_kmh'] for s in player_shots])), len(player_shots), player_shots[-1]['player_side'])
        return averages
