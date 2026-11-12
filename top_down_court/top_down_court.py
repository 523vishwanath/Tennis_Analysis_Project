import cv2
import numpy as np
import constants

# colors (BGR)
COLOR_IN_MY_SIDE = (0, 255, 0)           # green
COLOR_IN_OTHER_SIDE = (0, 255, 255)      # neon yellow
COLOR_OUT = (0, 0, 255)                  # red
COLOR_NET_TAPE = (255, 255, 255)         # white
COLOR_NET_BASE = (255, 0, 255)           # magenta
COLOR_NET_HIT = (255, 0, 255)            # magenta
COLOR_BALL_BOX = (0, 165, 255)           # orange
COLOR_PLAYER = (255, 200, 0)             # light blue
COLOR_SPEED_BOX = (40, 40, 40)           # dark grey box behind the speed text


class TopDownCourt:
    """Draws a flat top-down court (top right corner) plus bounces / net hits, and writes the output video."""

    def __init__(self, court_homography, pixels_per_meter=12, margin_meters=3, padding=20):
        self.court = court_homography
        self.pixels_per_meter = pixels_per_meter
        self.margin_meters = margin_meters
        self.padding = padding
        self.court_image = self.draw_court()

    # ---------------- helpers ----------------
    def court_to_mini(self, court_x, court_y):
        return (int((court_x + self.margin_meters) * self.pixels_per_meter),
                int((court_y + self.margin_meters) * self.pixels_per_meter))

    def get_bounce_color(self, bounce):
        if not bounce['is_in']:
            return COLOR_OUT
        if bounce['side'] == constants.MY_SIDE:
            return COLOR_IN_MY_SIDE
        return COLOR_IN_OTHER_SIDE

    def draw_x(self, image, center, size, color, thickness):
        x, y = center
        cv2.line(image, (x - size, y - size), (x + size, y + size), color, thickness)
        cv2.line(image, (x - size, y + size), (x + size, y - size), color, thickness)

    # ---------------- court drawing ----------------
    def draw_court(self):
        court_width = constants.DOUBLE_LINE_WIDTH
        court_length = constants.COURT_LENGTH
        alley = constants.DOUBLE_ALLY_DIFFERENCE
        service = constants.NO_MANS_LAND_HEIGHT

        canvas_width = int((court_width + 2 * self.margin_meters) * self.pixels_per_meter)
        canvas_height = int((court_length + 2 * self.margin_meters) * self.pixels_per_meter)
        court_image = np.full((canvas_height, canvas_width, 3), (60, 120, 60), dtype=np.uint8)

        court_lines = [
            ((0, 0), (court_width, 0)),                                          # far baseline
            ((0, court_length), (court_width, court_length)),                    # near baseline
            ((0, 0), (0, court_length)),                                         # left doubles
            ((court_width, 0), (court_width, court_length)),                     # right doubles
            ((alley, 0), (alley, court_length)),                                 # left singles
            ((court_width - alley, 0), (court_width - alley, court_length)),     # right singles
            ((alley, service), (court_width - alley, service)),                  # far service line
            ((alley, court_length - service), (court_width - alley, court_length - service)),  # near service line
            ((court_width / 2, service), (court_width / 2, court_length - service)),           # center service line
        ]
        for start, end in court_lines:
            cv2.line(court_image, self.court_to_mini(*start), self.court_to_mini(*end), (255, 255, 255), 1)

        # outline the singles court (safe landing area) a bit thicker
        cv2.rectangle(court_image, self.court_to_mini(constants.SINGLES_LEFT, 0),
                      self.court_to_mini(constants.SINGLES_RIGHT, court_length), (255, 255, 255), 2)

        # net
        cv2.line(court_image, self.court_to_mini(-constants.NET_POST_OUTSIDE_DOUBLES, court_length / 2),
                 self.court_to_mini(court_width + constants.NET_POST_OUTSIDE_DOUBLES, court_length / 2), (200, 200, 200), 2)
        return court_image

    # ---------------- per frame drawing ----------------
    def draw_frame(self, frame, frame_num, fps, raw_ball_dict, player_dict, bounces_so_far, net_hits_so_far, shots_so_far=None):

        # ---- main video ----
        self.court.draw_net(frame, COLOR_NET_TAPE, COLOR_NET_BASE)

        if 1 in raw_ball_dict:
            x1, y1, x2, y2 = raw_ball_dict[1]
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), COLOR_BALL_BOX, 2)

        for bounce in bounces_so_far:
            center = (int(bounce['image_x']), int(bounce['image_y']))
            cv2.circle(frame, center, 7, self.get_bounce_color(bounce), -1)
            cv2.circle(frame, center, 7, (0, 0, 0), 1)   # thin outline
        for net_hit in net_hits_so_far:
            self.draw_x(frame, (int(net_hit['image_x']), int(net_hit['image_y'])), 10, COLOR_NET_HIT, 3)

        # big label for 1 second after a new event
        if bounces_so_far and frame_num - bounces_so_far[-1]['frame'] < fps:
            last_bounce = bounces_so_far[-1]
            cv2.putText(frame, f"BOUNCE {last_bounce['label']}", (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.5, self.get_bounce_color(last_bounce), 3)
            if last_bounce.get('point_over'):
                cv2.putText(frame, "POINT OVER", (10, 300), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
        if net_hits_so_far and frame_num - net_hits_so_far[-1]['frame'] < fps:
            cv2.putText(frame, "NET", (10, 250), cv2.FONT_HERSHEY_SIMPLEX, 1.5, COLOR_NET_HIT, 3)

        # ---- top down court ----
        court_image = self.court_image.copy()

        for track_id, bbox in player_dict.items():
            x1, y1, x2, y2 = bbox
            feet_x, feet_y = self.court.image_to_court((x1 + x2) / 2, y2)
            cv2.circle(court_image, self.court_to_mini(feet_x, feet_y), 5, COLOR_PLAYER, -1)

        for bounce in bounces_so_far:
            cv2.circle(court_image, self.court_to_mini(bounce['court_x'], bounce['court_y']), 4, self.get_bounce_color(bounce), -1)
        if bounces_so_far:
            last_bounce = bounces_so_far[-1]
            last_point = self.court_to_mini(last_bounce['court_x'], last_bounce['court_y'])
            cv2.circle(court_image, last_point, 8, (255, 255, 255), 1)
            cv2.putText(court_image, str(last_bounce['frame']), (last_point[0] + 8, last_point[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

        for net_hit in net_hits_so_far:
            self.draw_x(court_image, self.court_to_mini(net_hit['court_x'], net_hit['court_y']), 5, COLOR_NET_HIT, 2)

        court_height, court_width = court_image.shape[:2]
        x_start = frame.shape[1] - court_width - self.padding
        y_start = self.padding
        frame[y_start:y_start + court_height, x_start:x_start + court_width] = court_image

        # legend under the mini court
        legend = [("IN (my side)", COLOR_IN_MY_SIDE), ("IN (other side)", COLOR_IN_OTHER_SIDE), ("OUT", COLOR_OUT)]
        for k, (text, color) in enumerate(legend):
            y_text = y_start + court_height + 22 + k * 22
            cv2.circle(frame, (x_start + 8, y_text - 5), 6, color, -1)
            cv2.putText(frame, text, (x_start + 20, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

        # average shot speed per player, under the legend
        if shots_so_far:
            self.draw_average_speeds(frame, shots_so_far, x_start, y_start + court_height + 22 + len(legend) * 22 + 10)

        cv2.putText(frame, f"Frame: {frame_num}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        return frame

    def draw_average_speeds(self, frame, shots_so_far, x_right_anchor, y_top):
        # {player_id: [speeds]} for the shots that already landed
        speeds_by_player = {}
        side_by_player = {}
        for shot in shots_so_far:
            speeds_by_player.setdefault(shot['player_id'], []).append(shot['speed_kmh'])
            side_by_player[shot['player_id']] = shot['player_side']

        lines = ["Avg shot speed"]
        for player_id in sorted(speeds_by_player):
            speeds = speeds_by_player[player_id]
            lines.append(f"P{player_id} ({side_by_player[player_id]}): {sum(speeds) / len(speeds):.0f} km/h  ({len(speeds)})")

        box_width = 290
        line_height = 26
        box_height = line_height * len(lines) + 12
        x_left = min(x_right_anchor, frame.shape[1] - box_width - self.padding)

        # semi transparent dark box
        overlay = frame.copy()
        cv2.rectangle(overlay, (x_left, y_top), (x_left + box_width, y_top + box_height), COLOR_SPEED_BOX, -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        for k, text in enumerate(lines):
            scale, thickness = (0.6, 2) if k == 0 else (0.55, 1)
            cv2.putText(frame, text, (x_left + 10, y_top + 24 + k * line_height), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), thickness)

    # ---------------- whole video ----------------
    def create_output_video(self, input_video_path, output_video_path, raw_ball_detections, player_detections, ball_bounces, net_hits, shots=None, progress=None):

        # read and write frame by frame, so the whole video is never in memory
        cap = cv2.VideoCapture(input_video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        codec = "mp4v" if output_video_path.lower().endswith(".mp4") else "MJPG"
        out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*codec), fps, (width, height))

        bounces_by_frame = {b['frame']: b for b in ball_bounces}
        net_hits_by_frame = {n['frame']: n for n in net_hits}
        bounces_so_far = []
        net_hits_so_far = []
        shots = shots or []

        frame_num = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_num in bounces_by_frame:
                bounces_so_far.append(bounces_by_frame[frame_num])
            if frame_num in net_hits_by_frame:
                net_hits_so_far.append(net_hits_by_frame[frame_num])

            # a shot counts once its ball has landed
            shots_so_far = [s for s in shots if s['bounce_frame'] <= frame_num]

            frame = self.draw_frame(frame, frame_num, fps, raw_ball_detections[frame_num], player_detections[frame_num],
                                    bounces_so_far, net_hits_so_far, shots_so_far)
            out.write(frame)
            frame_num += 1
            if progress is not None and frame_num % 10 == 0:
                progress("Drawing the output video", min(frame_num / len(raw_ball_detections), 1.0))

        cap.release()
        out.release()