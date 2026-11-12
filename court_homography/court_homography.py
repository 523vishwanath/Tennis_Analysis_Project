import cv2
import numpy as np
import constants
from .net_detector import find_net_tape, detect_net_bottom, refine_posts_with_mesh, snap_posts, tape_y
from .court_fitter import refine_court_points


class CourtHomography:
    """
    Maps image pixels <-> court meters using the 4 outer court corners, and knows where the net is.

    Court coordinates (meters): x = 0 .. 10.97 left to right, y = 0 (far baseline) .. 23.76 (near baseline),
    z = height above the ground.

    The homography uses keypoints 8, 9, 10, 11: where the 2 service lines meet the singles sidelines.
    That rectangle stays in the frame even when a baseline corner is cut off, and the whole court
    (baselines, doubles lines, net) is built from it.

    The bottom of the net comes from the homography (correct at any camera angle).
    The TOP of the net (tape) is found, best first:
      1. in the image itself: the white tape line, searched in a band above the net bottom (refine_net_with_image)
      2. from the 3D camera recovered from the court corners, only if that camera is stable
      3. straight up the screen (old method, fine for a level camera)
    """

    def __init__(self, court_keypoints, frame_size=None):
        # frame_size = (width, height) of the video, needed to recover the 3D camera
        self.court_width = constants.DOUBLE_LINE_WIDTH
        self.court_length = constants.COURT_LENGTH

        # keypoints 8-11: service line x singles sideline (the keypoint list is x0, y0, x1, y1, ...)
        singles_left = constants.SINGLES_LEFT                              # 1.37 m
        singles_right = constants.SINGLES_RIGHT                            # 9.60 m
        far_service_line = constants.NO_MANS_LAND_HEIGHT                   # 5.48 m from the far baseline
        near_service_line = self.court_length - constants.NO_MANS_LAND_HEIGHT   # 18.28 m

        image_points = np.array([
            [court_keypoints[16], court_keypoints[17]],   # point 8:  far left
            [court_keypoints[18], court_keypoints[19]],   # point 9:  far right
            [court_keypoints[20], court_keypoints[21]],   # point 10: near left
            [court_keypoints[22], court_keypoints[23]],   # point 11: near right
        ], dtype=np.float32)
        self.court_points = np.array([
            [singles_left, far_service_line],
            [singles_right, far_service_line],
            [singles_left, near_service_line],
            [singles_right, near_service_line],
        ], dtype=np.float32)

        self.court_keypoints = np.array(court_keypoints, dtype=float)
        self.frame_size = frame_size
        self.set_image_points(image_points)

    def set_image_points(self, image_points):
        # (re)build everything from where points 8-11 are in the image
        self.image_points = np.array(image_points, dtype=np.float32)
        self.court_keypoints[16:24] = self.image_points.reshape(-1)
        self.homography_matrix = cv2.getPerspectiveTransform(self.image_points, self.court_points)   # image -> court
        self.inverse_homography = np.linalg.inv(self.homography_matrix)                               # court -> image
        self.camera = self.estimate_camera(self.frame_size) if self.frame_size is not None else None
        if self.camera is not None and not self.camera_is_stable():
            self.camera = None                 # small keypoint errors would move the net a lot, don't trust it
        self.net = self.get_net_position()

    def refine_court_with_lines(self, frames, debug=False):
        """
        Move points 8-11 so the court lines sit on the real white lines (fixes keypoint errors).
        Run AFTER refine_net_with_image: the net bottom found in the image anchors the court's net line.
        """
        found_net = self.net if self.net['method'].startswith('found in the image') else None
        net_bottom = None
        if found_net is not None and 'bottom_line' in found_net:
            intercept, slope = found_net['bottom_line']
            net_bottom = (intercept, slope, found_net['left_post_bottom'][0], found_net['right_post_bottom'][0])

        better = refine_court_points(frames, self.image_points, self.court_points,
                                     exclude_polygon=self.net['zone_polygon'], net_bottom=net_bottom, debug=debug)
        if better is None:
            return False
        if debug:
            moved = np.linalg.norm(better - self.image_points, axis=1)
            print(f"    points 8-11 moved by {', '.join(f'{m:.0f}' for m in moved)} px")
        self.set_image_points(better)
        if found_net is not None:
            self.net = found_net          # keep the net found in the image, not the geometry guess
        return True

    # ---------------- conversions ----------------
    def image_to_court(self, x, y):
        # pixel -> meters (only correct for points ON the ground)
        return cv2.perspectiveTransform(np.array([[[x, y]]], dtype=np.float32), self.homography_matrix)[0][0]

    def court_to_image(self, x_meters, y_meters):
        # meters on the ground -> pixel
        return cv2.perspectiveTransform(np.array([[[x_meters, y_meters]]], dtype=np.float32), self.inverse_homography)[0][0]

    def court_3d_to_image(self, x_meters, y_meters, z_meters):
        # any 3D point (with height) -> pixel, using the recovered camera
        K, R, t = self.camera['K'], self.camera['R'], self.camera['t']
        p = K @ (R @ np.array([x_meters, y_meters, z_meters]) + t)
        return np.array([p[0] / p[2], p[1] / p[2]])

    # ---------------- 3D camera from the court ----------------
    def estimate_camera(self, frame_size):
        """
        The ground homography H (court -> image) = K [r1 r2 t] up to scale, where K has focal length f and the
        image center as optical center. r1 and r2 (the court's x and y directions) must be perpendicular and the
        same length, which gives f. Then r3 = r1 x r2 is the real "up" direction in the camera.
        """
        width, height = frame_size
        cx, cy = width / 2.0, height / 2.0

        # move the optical center to (0, 0) so K becomes diag(f, f, 1)
        shift = np.array([[1, 0, -cx], [0, 1, -cy], [0, 0, 1]], dtype=float)
        H = shift @ self.inverse_homography
        H = H / np.linalg.norm(H[:, 0])
        h1, h2, h3 = H[:, 0], H[:, 1], H[:, 2]

        # 2 ways to get f^2 (perpendicular + same length); use the ones that make sense
        estimates = []
        denominator = h1[2] * h2[2]
        if abs(denominator) > 1e-12:
            f2 = -(h1[0] * h2[0] + h1[1] * h2[1]) / denominator
            if f2 > 0:
                estimates.append(f2)
        denominator = h1[2] ** 2 - h2[2] ** 2
        if abs(denominator) > 1e-12:
            f2 = -(h1[0] ** 2 + h1[1] ** 2 - h2[0] ** 2 - h2[1] ** 2) / denominator
            if f2 > 0:
                estimates.append(f2)
        if not estimates:
            return None
        f = float(np.sqrt(np.median(estimates)))

        # a believable lens: focal length between 0.3x and 20x the image width
        if not (0.3 * width <= f <= 20 * width):
            return None

        K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=float)
        K_inv_H = np.linalg.inv(K) @ self.inverse_homography
        scale = (np.linalg.norm(K_inv_H[:, 0]) + np.linalg.norm(K_inv_H[:, 1])) / 2
        r1, r2, t = K_inv_H[:, 0] / scale, K_inv_H[:, 1] / scale, K_inv_H[:, 2] / scale
        if t[2] < 0:                               # the court must be in front of the camera
            r1, r2, t = -r1, -r2, -t
        r3 = np.cross(r1, r2)

        # clean up to an exact rotation
        U, _, Vt = np.linalg.svd(np.column_stack([r1, r2, r3]))
        R = U @ Vt

        # "up" (z) must point toward the camera, which is above the ground
        camera_position = -R.T @ t
        if camera_position[2] < 0:
            R[:, 2] = -R[:, 2]
            camera_position = -R.T @ t

        return {'K': K, 'R': R, 't': t, 'focal_length': f, 'camera_position': camera_position}

    def camera_is_stable(self, tries=12):
        # wiggle the court corners by a few pixels (like keypoint errors) and see how much the net moves
        if self.frame_size is None:
            return False
        rng = np.random.default_rng(0)
        noise_px = 3.0 * self.frame_size[0] / 1280
        reference = self.get_net_position()['tape_points']
        net_height_px = np.median(np.linalg.norm(reference - self.get_net_position()['base_points'], axis=1))
        worst = 0.0
        for _ in range(tries):
            keypoints = self.court_keypoints.copy()
            keypoints[16:24] += rng.normal(0, noise_px, 8)            # wiggle points 8-11
            wiggled = CourtHomography(keypoints)                       # no frame size -> no recursion
            wiggled.camera = wiggled.estimate_camera(self.frame_size)
            if wiggled.camera is None:
                return False
            worst = max(worst, np.abs(wiggled.get_net_position()['tape_points'] - reference).max())
        return worst < 0.2 * net_height_px

    # ---------------- net ----------------
    def net_height_at(self, x_meters):
        # 1.07 m at the posts, 0.914 m in the middle, straight in between
        post_x = -constants.NET_POST_OUTSIDE_DOUBLES
        center_x = self.court_width / 2
        ratio = min(abs(x_meters - center_x) / (center_x - post_x), 1.0)
        return constants.COURT_NET_HEIGHT + ratio * (constants.NET_HEIGHT_AT_POST - constants.COURT_NET_HEIGHT)

    def get_net_position(self):
        net_y = self.court_length / 2
        xs = np.linspace(-constants.NET_POST_OUTSIDE_DOUBLES, self.court_width + constants.NET_POST_OUTSIDE_DOUBLES, 21)

        base_points = np.array([self.court_to_image(x, net_y) for x in xs])

        if self.camera is not None:
            # real 3D net projected through the camera (works for any camera angle)
            tape_points = np.array([self.court_3d_to_image(x, net_y, self.net_height_at(x)) for x in xs])
            method = '3d camera'
        else:
            # fallback: the old way, net straight up the screen, 1 m = court width in px / 10.97 at the net
            pixels_per_meter = np.linalg.norm(self.court_to_image(self.court_width, net_y) - self.court_to_image(0, net_y)) / self.court_width
            tape_points = np.array([base_points[k] - [0, self.net_height_at(x) * pixels_per_meter] for k, x in enumerate(xs)])
            method = 'straight up (no camera)'

        return {
            'court_xs': xs,                        # court x (m) of every point below
            'base_points': base_points,            # 21 points along the bottom of the net, post to post
            'tape_points': tape_points,            # 21 points along the top (tape)
            'left_post_bottom': base_points[0],
            'right_post_bottom': base_points[-1],
            'center_bottom': base_points[len(xs) // 2],
            'left_post_top': tape_points[0],
            'right_post_top': tape_points[-1],
            'center_top': tape_points[len(xs) // 2],
            'zone_polygon': np.vstack([base_points, tape_points[::-1]]).astype(np.float32),
            'method': method,
        }

    def refine_net_with_image(self, frames, debug=False):
        """
        Find the real net in the video frames (frames = a few frames spread over the video):
          tape   = white line on top (detect_net_tape)
          posts  = where the tape starts and ends
          bottom = where the mesh / shadow of the net ends (detect_net_bottom)
        Only the parts that are found replace the geometry guess.
        """
        tape = find_net_tape(frames, self.net['base_points'], self.net['tape_points'], self.net['court_xs'],
                               alleys=[(0, constants.DOUBLE_ALLY_DIFFERENCE),
                                       (self.court_width - constants.DOUBLE_ALLY_DIFFERENCE, self.court_width)],
                               debug=debug)
        if tape is None:
            return False

        # the posts: where the net mesh under the tape stops
        rough_height = float(np.median(np.linalg.norm(self.net['base_points'] - self.net['tape_points'], axis=1)))
        tape = refine_posts_with_mesh(frames, tape, rough_height, debug=debug)

        # 21 points from post to post, along the found tape
        xs = np.linspace(tape['left_x'], tape['right_x'], 21)
        tape_points = np.column_stack([xs, tape_y(tape['coeffs'], xs, tape['center_x'])])
        bottom = detect_net_bottom(frames, tape, rough_height)
        if bottom is not None:
            # with tape and bottom known, snap the posts onto the actual poles
            tape = snap_posts(frames, tape, bottom, debug=debug)
            xs = np.linspace(tape['left_x'], tape['right_x'], 21)
            tape_points = np.column_stack([xs, tape_y(tape['coeffs'], xs, tape['center_x'])])
            intercept, slope = bottom
            base_points = np.column_stack([xs, intercept + slope * xs])
            self.net['bottom_line'] = (intercept, slope)
            method = 'found in the image (tape, posts, bottom)'
        else:
            # bottom not visible: keep the geometry guess, but only between the real posts
            base_points = np.column_stack([xs, [self.net_base_y_at(x) for x in xs]])
            method = 'found in the image (tape, posts), bottom from court keypoints'
        if debug:
            print(f"    net bottom: {'found' if bottom is not None else 'not found, using the keypoints'}")

        net = self.net
        net['tape_points'] = tape_points
        net['base_points'] = base_points
        net['left_post_top'], net['right_post_top'] = tape_points[0], tape_points[-1]
        net['left_post_bottom'], net['right_post_bottom'] = base_points[0], base_points[-1]
        net['center_top'], net['center_bottom'] = tape_points[10], base_points[10]
        net['zone_polygon'] = np.vstack([base_points, tape_points[::-1]]).astype(np.float32)
        net['method'] = method
        return True

    def _y_along(self, points, x):
        order = np.argsort(points[:, 0])
        return float(np.interp(x, points[order, 0], points[order, 1]))

    def net_top_y_at(self, x):
        # image y of the net tape at image x
        return self._y_along(self.net['tape_points'], x)

    def net_base_y_at(self, x):
        # image y of the bottom of the net at image x
        return self._y_along(self.net['base_points'], x)

    def is_in_net_zone(self, ball_x, ball_y, margin_px=15):
        # inside the net (between the posts, bottom to tape), or within margin_px of it
        distance = cv2.pointPolygonTest(self.net['zone_polygon'], (float(ball_x), float(ball_y)), True)
        return distance >= -margin_px

    def draw_net(self, frame, tape_color=(255, 255, 255), base_color=(255, 0, 255)):
        net = self.net
        cv2.polylines(frame, [net['tape_points'].astype(np.int32).reshape(-1, 1, 2)], False, tape_color, 2)
        cv2.polylines(frame, [net['base_points'].astype(np.int32).reshape(-1, 1, 2)], False, base_color, 2)
        for bottom, top in ((net['left_post_bottom'], net['left_post_top']), (net['right_post_bottom'], net['right_post_top'])):
            cv2.line(frame, (int(bottom[0]), int(bottom[1])), (int(top[0]), int(top[1])), tape_color, 2)   # posts
        return frame
