from utils import get_video_info, read_frames_at, get_device
import os
import time
import pickle
import cv2
from trackers import (PlayerTracker,
                      BallTracker,
                      WASBBallTracker)
from court_line_detector import CourtLineDetector
from court_homography import CourtHomography
from ball_events import BallEventDetector, ShotSpeedCalculator
from top_down_court import TopDownCourt
from trackers.ball_candidates import choose_moving_ball
import constants
import numpy as np

READ_FROM_STUB = False      # True = reuse the saved detections (much faster), False = run YOLO again
PLAYER_MODEL_PATH = 'yolo26l.pt'    # 'yolo26x.pt' is a bit more accurate but ~2x slower; players are easy to see
PLAYER_STUB_PATH = "tracker_stubs/player_detections.pkl"
BALL_STUB_PATH = "tracker_stubs/ball_candidates.pkl"
USE_WASB = True             # True = WASB tennis model (3 frames at once, finds blurry fast balls), False = old YOLO
WASB_MODEL_PATH = 'wasb/pretrained_weights/wasb_tennis_best.pth.tar'


def detect_players_and_balls(video_path, player_tracker, ball_tracker):
    # one frame at a time: the whole video is never in memory (1470 full HD frames = ~9 GB)
    fps, frame_count, width, height = get_video_info(video_path)
    cap = cv2.VideoCapture(video_path)
    player_detections, ball_candidates = [], []
    start = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        player_detections.append(player_tracker.detect_frame(frame))
        ball_candidates.append(ball_tracker.detect_frame_candidates(frame))

        done = len(player_detections)
        if done % 50 == 0 or done == frame_count:
            elapsed = time.time() - start
            per_frame = elapsed / done
            remaining = per_frame * (frame_count - done)
            print(f"  detecting: {done}/{frame_count} frames  ({1 / per_frame:.1f} frames/s, about {remaining / 60:.1f} min left)")
    cap.release()

    os.makedirs(os.path.dirname(PLAYER_STUB_PATH), exist_ok=True)
    with open(PLAYER_STUB_PATH, 'wb') as f:
        pickle.dump(player_detections, f)
    with open(BALL_STUB_PATH, 'wb') as f:
        pickle.dump(ball_candidates, f)
    return player_detections, ball_candidates


def main():
    # Read video frames
    input_video_path = "input_videos/input_video2.mp4"
    output_video_path = "output_videos_test/input_video2_new_ballweights.avi"
    fps, frame_count, frame_width, frame_height = get_video_info(input_video_path)
    print(f"Video: {frame_count} frames, {fps:.0f} fps, {frame_width}x{frame_height}, running on: {get_device()}")

    # Detect players and EVERY ball the model sees per frame
    player_tracker = PlayerTracker(model_path = PLAYER_MODEL_PATH)
    if USE_WASB:
        ball_tracker = WASBBallTracker(model_path = WASB_MODEL_PATH)
    else:
        ball_tracker = BallTracker(model_path = 'models/yolo5_last.pt')
    if READ_FROM_STUB and os.path.exists(PLAYER_STUB_PATH) and os.path.exists(BALL_STUB_PATH):
        print("Using saved detections")
        with open(PLAYER_STUB_PATH, 'rb') as f:
            player_detections = pickle.load(f)
        with open(BALL_STUB_PATH, 'rb') as f:
            ball_candidates = pickle.load(f)
    else:
        player_detections, ball_candidates = detect_players_and_balls(input_video_path, player_tracker, ball_tracker)

    # keep only the MOVING ball (balls lying on the court are ignored)
    raw_ball_detections, stationary_balls = choose_moving_ball(ball_candidates, fps, frame_height, debug=True)
    print(f"Moving ball found in {sum(1 in d for d in raw_ball_detections)}/{len(raw_ball_detections)} frames")

    # interpolated copy is only for the old shot detection, bounce/net use the RAW detections
    ball_detections = ball_tracker.interpolate_ball_positions(raw_ball_detections)

    #Court line detector model
    court_line_detector = CourtLineDetector(model_path='models/keypoints_model.pth')
    first_frame = read_frames_at(input_video_path, [0])[0]
    court_keypoints = court_line_detector.predict(first_frame)

    # a few frames spread over the video, to find the net tape with the players out of the way
    net_frames = read_frames_at(input_video_path, np.linspace(0, frame_count - 1, 9).astype(int))

    # Court homography from keypoints 8-11 (service lines x singles lines), the whole court is built from it
    court_homography = CourtHomography(court_keypoints, frame_size=(frame_width, frame_height))
    court_homography.refine_net_with_image(net_frames, debug=True)
    print("Net position:", court_homography.net['method'])

    # fix keypoint errors: move points 8-11 until the court lines sit on the real white lines
    # (the net found above anchors it). This is what puts bounces in the right place on the mini court.
    if court_homography.refine_court_with_lines(net_frames, debug=True):
        print("Court: corrected with the white court lines")
    else:
        print("Court: using the keypoints as they are")
    del net_frames

    # choosing the 2 players by where they stand on the court (P1 = near, P2 = far)
    player_detections = player_tracker.choose_players_by_court(player_detections, court_homography)

    #Detect ball shots
    ball_shot_frames, ball_hit_this_side, ball_hit_other_side = ball_tracker.get_ball_shot_frames(ball_detections)
    print("Ball shot frames:", ball_shot_frames)

    # Bounces (with IN / OUT) and net hits
    ball_event_detector = BallEventDetector(court_homography, fps=fps, frame_height=frame_height, debug=True)
    ball_bounce_frames, ball_bounces = ball_event_detector.get_ball_bounces(raw_ball_detections, player_detections)
    ball_bounce_frames, ball_bounces = ball_event_detector.filter_same_side_bounces(ball_bounces, ball_shot_frames,
                                                                                   ball_detections, player_detections)
    net_hit_frames, net_hits = ball_event_detector.get_net_hits(raw_ball_detections, player_detections,
                                                                ball_shot_frames, ball_detections, ball_bounces)

    print("Ball bounce frames:", ball_bounce_frames)
    for b in ball_bounces:
        side_text = 'my side' if b['side'] == constants.MY_SIDE else 'other side'
        out_text = f" ({b['distance_out']:.2f} m out)" if not b['is_in'] else ''
        over_text = "  -> POINT OVER (double bounce)" if b['point_over'] else ''
        print(f"  bounce {b['frame']}: court ({b['court_x']:.1f} m, {b['court_y']:.1f} m)  {side_text}  {b['label']}{out_text}{over_text}")
    print("Net hit frames:", net_hit_frames)
    for n in net_hits:
        print(f"  net {n['frame']}: P{n['player_id']} ({n['player_side']}), {n['type']}, {n['reason']}")

    # Shot speeds: hitter's feet -> bounce, divided by the flight time
    shot_speed_calculator = ShotSpeedCalculator(court_homography, fps)
    shots = shot_speed_calculator.get_shot_speeds(ball_shot_frames, ball_detections, player_detections, ball_bounces)

    print("Shots:")
    for s in shots:
        print(f"  P{s['player_id']} ({s['player_side']}) hit {s['hit_frame']} -> bounce {s['bounce_frame']}: "
              f"{s['distance_meters']:.1f} m in {s['flight_seconds']:.2f} s = {s['speed_kmh']:.0f} km/h")
    for player_id, (average_kmh, count, side) in ShotSpeedCalculator.get_average_speeds(shots).items():
        print(f"  P{player_id} ({side}) average: {average_kmh:.0f} km/h over {count} shots")

    # Draw everything and save the video (same fps as the input)
    top_down_court = TopDownCourt(court_homography)
    top_down_court.create_output_video(input_video_path, output_video_path, raw_ball_detections,
                                       player_detections, ball_bounces, net_hits, shots)
    print("Saved", output_video_path)


if __name__ == "__main__":
    main()