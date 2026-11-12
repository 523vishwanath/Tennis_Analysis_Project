import numpy as np


def get_scales(fps, frame_height):
    # settings below are written for a 60 fps, 1080p video and scaled for any other video
    time_scale = fps / 60.0
    pixel_scale = frame_height / 1080.0
    return time_scale, pixel_scale


def find_stationary_balls(ball_candidates, fps, frame_height):
    """
    A ball that is detected in the SAME spot (within a few pixels) for 1 second or more is lying on the court.
    Returns the spots: {'x', 'y', 'first_frame', 'last_frame', 'frames'}
    """
    time_scale, pixel_scale = get_scales(fps, frame_height)
    same_spot_px = 8 * pixel_scale                 # detections closer than this = same spot (detector jitter)
    min_frames = int(60 * time_scale)              # seen there for at least 1 s (a flying ball never is)
    max_missing = int(30 * time_scale)             # the spot can disappear for 0.5 s (hidden by a player) and still be the same ball

    spots = []                                     # every spot a ball was seen
    for frame_num, candidates in enumerate(ball_candidates):
        for x1, y1, x2, y2, confidence in candidates:
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            match = None
            for spot in spots:
                close = np.hypot(cx - spot['x'], cy - spot['y']) <= same_spot_px
                recent = frame_num - spot['last_frame'] <= max_missing
                if close and recent:
                    match = spot
                    break
            if match is None:
                spots.append({'x': cx, 'y': cy, 'first_frame': frame_num, 'last_frame': frame_num, 'frames': 1})
            else:
                # running average keeps the spot centered on the ball
                match['x'] = (match['x'] * match['frames'] + cx) / (match['frames'] + 1)
                match['y'] = (match['y'] * match['frames'] + cy) / (match['frames'] + 1)
                match['last_frame'] = frame_num
                match['frames'] += 1

    return [spot for spot in spots if spot['frames'] >= min_frames]


def choose_moving_ball(ball_candidates, fps, frame_height, debug=False):
    """
    ball_candidates: per frame, a list of [x1, y1, x2, y2, confidence] (all balls the model saw)
    Returns per frame {1: [x1, y1, x2, y2]} for the MOVING ball (or {} if there is none),
    the same format as before, so everything after this step works unchanged.
    """
    time_scale, pixel_scale = get_scales(fps, frame_height)
    ignore_radius_px = 15 * pixel_scale            # candidates this close to a lying ball are that ball
    max_ball_speed_px = 45 * pixel_scale / time_scale   # px per frame, faster than any real shot
    lost_after_frames = int(30 * time_scale)       # no ball for 0.5 s -> start fresh with the best candidate
    max_off_path_px = 25 * pixel_scale / time_scale     # px per frame a real ball can drift from its predicted path

    stationary_balls = find_stationary_balls(ball_candidates, fps, frame_height)

    ball_detections = []
    history = []                                   # (frame_num, cx, cy) of the chosen ball
    ignored = 0

    for frame_num, candidates in enumerate(ball_candidates):

        # 1. drop balls lying on the court (only while they are actually lying there)
        moving = []
        for x1, y1, x2, y2, confidence in candidates:
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            on_lying_ball = any(spot['first_frame'] <= frame_num <= spot['last_frame'] and
                                np.hypot(cx - spot['x'], cy - spot['y']) <= ignore_radius_px
                                for spot in stationary_balls)
            if on_lying_ball:
                ignored += 1
            else:
                moving.append((x1, y1, x2, y2, confidence, cx, cy))

        if not moving:
            ball_detections.append({})
            continue

        # 2. follow the ball: pick the candidate closest to where the ball should be now
        recent = [h for h in history if frame_num - h[0] <= lost_after_frames]
        if not recent:
            best = max(moving, key=lambda c: c[4])             # nothing to follow yet: most confident one
        else:
            last_frame, last_x, last_y = recent[-1]
            gap = frame_num - last_frame
            if len(recent) >= 2:
                prev_frame, prev_x, prev_y = recent[-2]
                step = max(last_frame - prev_frame, 1)
                predicted_x = last_x + (last_x - prev_x) / step * gap
                predicted_y = last_y + (last_y - prev_y) / step * gap
            else:
                predicted_x, predicted_y = last_x, last_y

            def distance_to_prediction(c):
                return np.hypot(c[5] - predicted_x, c[6] - predicted_y)

            best = min(moving, key=distance_to_prediction)

            def reachable(point):
                # could a real ball get from this earlier point to the candidate in time?
                point_frame, point_x, point_y = point
                return np.hypot(best[5] - point_x, best[6] - point_y) <= max_ball_speed_px * (frame_num - point_frame)

            def on_path_skipping_last():
                # does the candidate continue the path the ball had BEFORE the last point?
                if len(recent) < 3:
                    return reachable(recent[-2])
                (f1, x1_, y1_), (f2, x2_, y2_) = recent[-3], recent[-2]
                gap2 = frame_num - f2
                predicted = (x2_ + (x2_ - x1_) / max(f2 - f1, 1) * gap2, y2_ + (y2_ - y1_) / max(f2 - f1, 1) * gap2)
                return reachable(recent[-2]) and np.hypot(best[5] - predicted[0], best[6] - predicted[1]) <= max_off_path_px * gap2

            if not reachable(recent[-1]):
                if len(recent) >= 2 and on_path_skipping_last():
                    # it fits the ball from 2 points ago -> the LAST point was a one-frame stray detection, remove it
                    stray_frame = recent[-1][0]
                    ball_detections[stray_frame] = {}
                    history.remove(recent[-1])
                else:
                    # too far for a real ball = a false detection somewhere else
                    ball_detections.append({})
                    continue

        x1, y1, x2, y2, confidence, cx, cy = best
        ball_detections.append({1: [x1, y1, x2, y2]})
        history.append((frame_num, cx, cy))

    if debug or stationary_balls:
        print(f"  stationary balls ignored: {len(stationary_balls)} "
              + ", ".join(f"at ({s['x']:.0f}, {s['y']:.0f}) frames {s['first_frame']}-{s['last_frame']}" for s in stationary_balls))

    return ball_detections, stationary_balls


def ball_detections_to_candidates(ball_detections):
    # old stubs only kept 1 box per frame -> turn them into the candidate format
    return [[list(d[1]) + [1.0]] if 1 in d else [] for d in ball_detections]
