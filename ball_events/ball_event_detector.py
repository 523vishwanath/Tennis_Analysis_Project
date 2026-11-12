import numpy as np
import pandas as pd
import constants
from utils import is_close_to_player, fit_curve, fit_error, curve_speed
from .hitter import get_hitter


class BallEventDetector:
    """Finds ball bounces (with IN/OUT) and net hits from the RAW ball detections."""

    def __init__(self, court_homography, fps=60.0, frame_height=1080, debug=False):
        self.court = court_homography
        self.debug = debug

        # All settings below were tuned on a 60 fps, 1080p video.
        # They are scaled so the same seconds / same real speeds are used for any fps and resolution.
        time_scale = fps / 60.0                       # 30 fps -> 0.5 (half as many frames per second)
        pixel_scale = frame_height / 1080.0           # 720p  -> 0.67 (everything is smaller)
        speed_scale = pixel_scale / time_scale        # px/frame: smaller image, but more movement per frame

        def frames(n):                                # n frames at 60 fps -> same time at this fps
            return max(1, int(round(n * time_scale)))

        # ---------------- bounce settings ----------------
        self.frames_window = frames(30)               # how far left/right of a point we look to fit the ball path (0.5 s)
        self.max_points_per_side = 12                 # max detections used on each side
        self.min_points_per_side = 3                  # min detections needed on each side
        self.max_gap_frames = frames(12)              # bigger gap than this between 2 detections = can't place a bounce (0.2 s)
        self.minimum_kink = 3.0 * speed_scale         # px/frame drop in y speed (falling -> rising)
        self.minimum_fit_gain = 3.0                   # 2 curves must fit at least 3x better than 1 curve
        self.court_margin_meters = 1.5                # bounce must be inside the court + this margin
        self.min_frames_between_bounces = frames(20)
        self.min_side_speed = 1.5 * speed_scale       # px/frame, sideways speed that counts for a direction flip
        self.min_reversal_speed = 2.0 * speed_scale   # px/frame, up/down speed that counts for a racket reversal
        self.outlier_jump_px = 50 * speed_scale       # one frame jump bigger than this = false detection
        self.net_zone_margin_px = 15 * pixel_scale

        # ---------------- net hit settings ----------------
        self.net_frames_window = frames(15)
        self.min_speed_before_net = 6.0 * speed_scale # px/frame, ball must be flying at the net
        self.max_speed_ratio_net = 0.35               # speed after must drop below 35% of speed before
        self.track_lost_frames = frames(30)           # ball vanishes for this long right after reaching the net
        self.min_frames_between_net_hits = frames(30)
        self.max_frames_since_hit = frames(90)        # the net hit must come within 1.5 s of a hit (to know who hit it)
        self.no_hit_frames_after_net = frames(60)     # after a net hit: no return from the other player for 1 s
        self.skip_frames_after_net = frames(5)        # ignore "hits" right at the net contact (the net bounce itself)
        self.point_over_pause_frames = frames(120)    # no real bounce for 2 s after a double bounce = the point is over

    # =====================================================================
    # Clean ball positions
    # =====================================================================
    def get_real_ball_positions(self, raw_ball_detections):
        ball_positions = [x.get(1,[np.nan]*4) for x in raw_ball_detections]
        df_ball_positions = pd.DataFrame(ball_positions,columns=['x1','y1','x2','y2'])
        df_ball_positions['mid_x'] = (df_ball_positions['x1'] + df_ball_positions['x2']) / 2
        df_ball_positions['mid_y'] = (df_ball_positions['y1'] + df_ball_positions['y2']) / 2
        df_ball_positions['ball_height'] = df_ball_positions['y2'] - df_ball_positions['y1']

        # remove one frame jumps (false detections)
        median_x = df_ball_positions['mid_x'].rolling(window =7,min_periods=1, center =True).median()
        median_y = df_ball_positions['mid_y'].rolling(window =7,min_periods=1, center =True).median()
        jump = np.sqrt((df_ball_positions['mid_x'] - median_x) ** 2 + (df_ball_positions['mid_y'] - median_y) ** 2)
        df_ball_positions.loc[jump > self.outlier_jump_px, ['mid_x', 'mid_y']] = np.nan

        # only REAL detections (no interpolation)
        return df_ball_positions.dropna(subset=['mid_x', 'mid_y'])

    # =====================================================================
    # Bounces
    # =====================================================================
    def get_ball_bounces(self, raw_ball_detections, player_detections):

        df_real = self.get_real_ball_positions(raw_ball_detections)
        frames = df_real.index.to_numpy().astype(float)
        mid_x = df_real['mid_x'].to_numpy()
        mid_y = df_real['mid_y'].to_numpy()
        ball_half_height = df_real['ball_height'].median() / 2

        court_width = self.court.court_width
        court_length = self.court.court_length

        #Logic to find bounce candidates between every 2 real detections
        candidates = []
        rejected = []

        for i in range(1, len(frames)):

            if frames[i] - frames[i - 1] > self.max_gap_frames:
                continue

            left = [j for j in range(i) if frames[i - 1] - frames[j] <= self.frames_window][-self.max_points_per_side:]
            right = [j for j in range(i, len(frames)) if frames[j] - frames[i] <= self.frames_window][:self.max_points_per_side]

            if len(left) < self.min_points_per_side or len(right) < self.min_points_per_side:
                continue

            t_left, t_right = frames[left], frames[right]

            # one curve before the point, one curve after it
            left_y, right_y = fit_curve(t_left, mid_y[left]), fit_curve(t_right, mid_y[right])
            left_x, right_x = fit_curve(t_left, mid_x[left]), fit_curve(t_right, mid_x[right])

            two_curve_error = (fit_error(left_y, t_left, mid_y[left]) + fit_error(right_y, t_right, mid_y[right]) +
                               fit_error(left_x, t_left, mid_x[left]) + fit_error(right_x, t_right, mid_x[right]))

            # one curve over everything (= no bounce)
            t_all = np.r_[t_left, t_right]
            y_all = np.r_[mid_y[left], mid_y[right]]
            x_all = np.r_[mid_x[left], mid_x[right]]
            one_curve_error = (fit_error(np.polyfit(t_all, y_all, 2), t_all, y_all) +
                               fit_error(np.polyfit(t_all, x_all, 2), t_all, x_all))

            fit_gain = one_curve_error / max(two_curve_error, 1)

            # exact bounce time = where the 2 curves meet (works even if the bounce frame itself was missed)
            t_search = np.linspace(frames[i - 1], frames[i], 50)
            bounce_time = t_search[np.argmin(np.abs(np.polyval(left_y, t_search) - np.polyval(right_y, t_search)))]
            bounce_frame = int(round(bounce_time))

            speed_y_before, speed_y_after = curve_speed(left_y, bounce_time), curve_speed(right_y, bounce_time)
            speed_x_before, speed_x_after = curve_speed(left_x, bounce_time), curve_speed(right_x, bounce_time)

            # + kink = ball was falling (down the screen) and suddenly started rising
            kink = speed_y_before - speed_y_after

            ball_x = (np.polyval(left_x, bounce_time) + np.polyval(right_x, bounce_time)) / 2
            ball_y = (np.polyval(left_y, bounce_time) + np.polyval(right_y, bounce_time)) / 2

            # bottom of the ball on the court in meters
            court_x, court_y = self.court.image_to_court(ball_x, ball_y + ball_half_height)

            close_to_player = is_close_to_player(player_detections[bounce_frame], ball_x, ball_y)

            # ---------------- physics checks ----------------
            # 1. hit: ball changes left/right direction (a bounce keeps its sideways direction)
            x_direction_flip = (np.sign(speed_x_before) != np.sign(speed_x_after)) and abs(speed_x_before) > self.min_side_speed and abs(speed_x_after) > self.min_side_speed
            # 2. hit: ball comes at a player and goes straight back the other way
            racket_reversal = speed_y_before > self.min_reversal_speed and speed_y_after < -self.min_reversal_speed and close_to_player
            # 3. serve toss / overhead: ball already rising past a player
            toss_or_hit = close_to_player and speed_y_before < 0
            # 4. a bounce has to happen on the court
            on_court = (-self.court_margin_meters <= court_x <= court_width + self.court_margin_meters and
                        -self.court_margin_meters <= court_y <= court_length + self.court_margin_meters)
            # 5. kink at the net tape = net touch, not a bounce (a bounce can't land on the net line)
            touches_net = self.court.is_in_net_zone(ball_x, ball_y, self.net_zone_margin_px) and abs(court_y - court_length / 2) < 0.5

            strong_bend = kink >= self.minimum_kink and fit_gain >= self.minimum_fit_gain
            passes_checks = on_court and not x_direction_flip and not racket_reversal and not toss_or_hit and not touches_net

            if strong_bend and not passes_checks:
                reasons = []
                if not on_court: reasons.append('off court')
                if x_direction_flip: reasons.append('sideways flip (hit)')
                if racket_reversal: reasons.append('reversal at player (hit)')
                if toss_or_hit: reasons.append('rising at player (toss/hit)')
                if touches_net: reasons.append('at net tape')
                rejected.append((bounce_frame, round(float(kink), 1), round(float(fit_gain), 1), ', '.join(reasons)))

            if strong_bend and passes_checks:
                candidates.append({
                    'frame': bounce_frame,
                    'score': kink * np.log(fit_gain),
                    'court_x': float(court_x),
                    'court_y': float(court_y),
                    'image_x': float(ball_x),
                    'image_y': float(ball_y + ball_half_height),
                })

        # keep only the strongest candidate when several are close together
        ball_bounces = []
        for candidate in sorted(candidates, key=lambda c: c['score'], reverse=True):
            if all(abs(candidate['frame'] - b['frame']) > self.min_frames_between_bounces for b in ball_bounces):
                ball_bounces.append(candidate)

        ball_bounces = sorted(ball_bounces, key=lambda c: c['frame'])
        ball_bounces = self.classify_bounces(ball_bounces)
        ball_bounce_frames = [b['frame'] for b in ball_bounces]

        if self.debug:
            print("Rejected strong candidates (frame, kink, fit gain, reason):")
            last_frame = -100
            for r in rejected:
                if r[0] - last_frame > 3:      # don't print the same event 5 times
                    print("  ", r)
                last_frame = r[0]
            gaps = [(int(frames[i - 1]), int(frames[i])) for i in range(1, len(frames)) if frames[i] - frames[i - 1] > self.max_gap_frames]
            print("Gaps with no ball detected (bounces here can't be found):", gaps)

        return ball_bounce_frames, ball_bounces

    # =====================================================================
    # Only ONE bounce per side until the ball crosses the net
    # =====================================================================
    def filter_same_side_bounces(self, ball_bounces, ball_shot_frames, ball_detections, player_detections):
        # ball_detections = INTERPOLATED (to find who hit the ball)
        # The ball crosses the net when the player on the OTHER side hits it.
        # So a 2nd bounce on the same side is only real if the other player hit the ball in between.

        # side of the player who made each hit
        hit_sides = {}
        for hit_frame in ball_shot_frames:
            hitter = get_hitter(self.court, hit_frame, ball_detections, player_detections)
            if hitter is not None:
                hit_sides[hit_frame] = hitter['side']

        kept_bounces = []
        discarded_bounces = []

        for bounce in sorted(ball_bounces, key=lambda b: b['frame']):
            bounce['point_over'] = False

            if not kept_bounces:
                kept_bounces.append(bounce)
                continue

            last_bounce = kept_bounces[-1]

            if bounce['side'] != last_bounce['side']:
                kept_bounces.append(bounce)          # landed on the other side = the ball crossed the net
                continue

            # same side as the last bounce: did the other player send the ball over in between?
            hits_between = [f for f in hit_sides if last_bounce['frame'] < f < bounce['frame']]
            crossed = any(hit_sides[f] != bounce['side'] for f in hits_between)

            if crossed:
                kept_bounces.append(bounce)
            else:
                # 2nd bounce without crossing: false bounce, throw it away
                last_bounce['had_extra_bounce'] = True
                discarded_bounces.append(bounce)

        # the point is over at a bounce that had extra bounces after it AND nobody played the ball after it:
        # the player on that side didn't hit it back, and there's no other real bounce soon after
        for k, bounce in enumerate(kept_bounces):
            if bounce.pop('had_extra_bounce', False):
                window_end = bounce['frame'] + self.point_over_pause_frames
                returned = any(bounce['frame'] < f <= window_end and side == bounce['side'] for f, side in hit_sides.items())
                next_bounce = kept_bounces[k + 1] if k + 1 < len(kept_bounces) else None
                rally_goes_on = next_bounce is not None and next_bounce['frame'] <= window_end
                bounce['point_over'] = not returned and not rally_goes_on

        if self.debug and discarded_bounces:
            print("Discarded bounces (2nd bounce on the same side, ball never crossed):",
                  [(b['frame'], b['side']) for b in discarded_bounces])

        return [b['frame'] for b in kept_bounces], kept_bounces

    # =====================================================================
    # IN / OUT (singles) + which side
    # =====================================================================
    def classify_bounces(self, ball_bounces):
        tolerance = constants.LINE_TOLERANCE_METERS

        for bounce in ball_bounces:
            court_x, court_y = bounce['court_x'], bounce['court_y']

            # side of the net the ball landed on
            bounce['side'] = 'far' if court_y < self.court.court_length / 2 else 'near'

            # inside the singles court? (touching the line = IN)
            inside_x = constants.SINGLES_LEFT - tolerance <= court_x <= constants.SINGLES_RIGHT + tolerance
            inside_y = 0 - tolerance <= court_y <= self.court.court_length + tolerance
            bounce['is_in'] = inside_x and inside_y
            bounce['label'] = 'IN' if bounce['is_in'] else 'OUT'

            # how far out (0 if in), useful for close calls
            out_x = max(constants.SINGLES_LEFT - court_x, court_x - constants.SINGLES_RIGHT, 0)
            out_y = max(0 - court_y, court_y - self.court.court_length, 0)
            bounce['distance_out'] = float(np.hypot(out_x, out_y))

        return ball_bounces

    # =====================================================================
    # Net hits
    # =====================================================================
    def get_net_hits(self, raw_ball_detections, player_detections, ball_shot_frames, ball_detections, ball_bounces):
        # ball_detections = INTERPOLATED (to find who hit the ball), ball_bounces = from get_ball_bounces()

        df_real = self.get_real_ball_positions(raw_ball_detections)
        frames = df_real.index.to_numpy().astype(float)
        mid_x = df_real['mid_x'].to_numpy()
        mid_y = df_real['mid_y'].to_numpy()

        #Logic to find net hits
        candidates = []

        for i in range(1, len(frames)):

            left = [j for j in range(i) if frames[i - 1] - frames[j] <= self.net_frames_window]
            if len(left) < self.min_points_per_side:
                continue

            left_x, left_y = fit_curve(frames[left], mid_x[left]), fit_curve(frames[left], mid_y[left])
            t_contact = frames[i - 1]
            ball_x, ball_y = np.polyval(left_x, t_contact), np.polyval(left_y, t_contact)

            if not self.court.is_in_net_zone(ball_x, ball_y, self.net_zone_margin_px):
                continue
            if is_close_to_player(player_detections[int(t_contact)], ball_x, ball_y):
                continue   # volley at the net, not a net hit

            speed_before = np.hypot(curve_speed(left_x, t_contact), curve_speed(left_y, t_contact))
            if speed_before < self.min_speed_before_net:
                continue

            next_gap = frames[i] - frames[i - 1]

            # case 1: ball reaches the net and the track ends (dropped into the net / behind the tape)
            if next_gap >= self.track_lost_frames:
                hit_type = 'net (ball lost at net)'
                speed_after = 0.0
            elif next_gap <= self.max_gap_frames:
                right = [j for j in range(i, len(frames)) if frames[j] - frames[i] <= self.net_frames_window]
                if len(right) < self.min_points_per_side:
                    continue
                right_x, right_y = fit_curve(frames[right], mid_x[right]), fit_curve(frames[right], mid_y[right])
                speed_after = np.hypot(curve_speed(right_x, frames[i]), curve_speed(right_y, frames[i]))
                # case 2: ball suddenly loses most of its speed at the net (a bounce or a flight never does this)
                if speed_after >= self.max_speed_ratio_net * speed_before:
                    continue
                hit_type = 'net (ball stopped)'
            else:
                continue

            # ---------------- confirm: the ball did NOT get to the other side ----------------
            contact_frame = int(t_contact)

            # who hit it? (last hit before the net contact)
            hits_before = [f for f in ball_shot_frames if contact_frame - self.max_frames_since_hit <= f < contact_frame - self.skip_frames_after_net]
            if not hits_before:
                continue        # don't know who hit it, so we can't tell which side it should stay on
            hitter = get_hitter(self.court, hits_before[-1], ball_detections, player_detections)
            if hitter is None:
                continue

            window_start = contact_frame + self.skip_frames_after_net
            window_end = contact_frame + self.no_hit_frames_after_net

            # hits in the next 60 frames, split by side
            hits_after = [f for f in ball_shot_frames if window_start < f <= window_end]
            hit_sides_after = [h['side'] for h in (get_hitter(self.court, f, ball_detections, player_detections) for f in hits_after) if h]
            other_player_hit = any(side != hitter['side'] for side in hit_sides_after)

            # bounces in the next 60 frames, split by side
            bounces_after = [b for b in ball_bounces if window_start < b['frame'] <= window_end]
            bounced_other_side = any(b['side'] != hitter['side'] for b in bounces_after)
            bounced_same_side = any(b['side'] == hitter['side'] for b in bounces_after)

            # crossed = the other player hit it back, or it landed on the other side
            if other_player_hit or bounced_other_side:
                continue

            # net hit = nobody hits it for 60 frames, OR it dropped back on the hitter's side
            if hits_after and not bounced_same_side:
                continue

            court_x, _ = self.court.image_to_court(ball_x, self.court.net_base_y_at(ball_x))   # where along the net
            candidates.append({
                'frame': int(t_contact),
                'type': hit_type,
                'speed_before': float(speed_before),
                'speed_after': float(speed_after),
                'image_x': float(ball_x),
                'image_y': float(ball_y),
                'court_x': float(court_x),
                'court_y': self.court.court_length / 2,
                'player_id': hitter['player_id'],
                'player_side': hitter['side'],
                'reason': 'dropped on own side' if bounced_same_side else 'no return for 1 s',
            })

        net_hits = []
        for candidate in sorted(candidates, key=lambda c: c['frame']):
            if all(abs(candidate['frame'] - n['frame']) > self.min_frames_between_net_hits for n in net_hits):
                net_hits.append(candidate)

        net_hit_frames = [n['frame'] for n in net_hits]
        return net_hit_frames, net_hits