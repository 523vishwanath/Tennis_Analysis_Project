import cv2
import numpy as np
from scipy.optimize import minimize
from scipy.ndimage import map_coordinates
import constants

COURT_WIDTH = constants.DOUBLE_LINE_WIDTH
COURT_LENGTH = constants.COURT_LENGTH
ALLEY = constants.DOUBLE_ALLY_DIFFERENCE
SERVICE = constants.NO_MANS_LAND_HEIGHT

# every painted court line, in meters
COURT_LINES = [
    ((0, 0), (COURT_WIDTH, 0)), ((0, COURT_LENGTH), (COURT_WIDTH, COURT_LENGTH)),                 # baselines
    ((0, 0), (0, COURT_LENGTH)), ((COURT_WIDTH, 0), (COURT_WIDTH, COURT_LENGTH)),                 # doubles sidelines
    ((ALLEY, 0), (ALLEY, COURT_LENGTH)), ((COURT_WIDTH - ALLEY, 0), (COURT_WIDTH - ALLEY, COURT_LENGTH)),   # singles
    ((ALLEY, SERVICE), (COURT_WIDTH - ALLEY, SERVICE)),                                           # far service line
    ((ALLEY, COURT_LENGTH - SERVICE), (COURT_WIDTH - ALLEY, COURT_LENGTH - SERVICE)),             # near service line
    ((COURT_WIDTH / 2, SERVICE), (COURT_WIDTH / 2, COURT_LENGTH - SERVICE)),                      # center service line
]


def court_line_samples(points_per_line=80):
    return np.vstack([np.linspace(a, b, points_per_line) for a, b in COURT_LINES]).astype(np.float32)


def white_line_distance_map(frames, exclude_polygon=None):
    """For every pixel: distance (px) to the nearest white court-line pixel."""
    image = frames[0] if len(frames) == 1 else np.median(np.stack(frames), axis=0).astype(np.uint8)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15)))
    white = (tophat > 25).astype(np.uint8)
    if exclude_polygon is not None:
        # the net tape is white too, but it's not a court line
        cv2.fillPoly(white, [exclude_polygon.astype(np.int32)], 0)
    return cv2.distanceTransform((1 - white).astype(np.uint8), cv2.DIST_L2, 5).astype(np.float32)


def fit_score(distance_map, image_points, court_points, samples, cap, net_bottom=None):
    """
    Average (capped) distance from the drawn court lines to the real white lines. Lower = better.
    net_bottom = the net's bottom line found in the image; the court's net line (y = 11.88 m) must lie on it.
    """
    H = cv2.getPerspectiveTransform(court_points, image_points.reshape(4, 2).astype(np.float32))
    projected = cv2.perspectiveTransform(samples.reshape(-1, 1, 2), H).reshape(-1, 2)
    h, w = distance_map.shape
    inside = (projected[:, 0] >= 1) & (projected[:, 0] < w - 1) & (projected[:, 1] >= 1) & (projected[:, 1] < h - 1)
    if inside.mean() < 0.25:
        return cap ** 2 * 10, inside.mean()                  # most of the court off-screen: not a real fit
    d = map_coordinates(distance_map, [projected[inside, 1], projected[inside, 0]], order=1)
    score = float(np.mean(np.minimum(d, cap) ** 2))
    on_line = float((d < 3).mean())

    if net_bottom is not None:
        # the court's net line, from post to post, must sit on the net bottom found in the image
        net_line = np.column_stack([np.linspace(0, COURT_WIDTH, 30), np.full(30, COURT_LENGTH / 2)]).astype(np.float32)
        net_pixels = cv2.perspectiveTransform(net_line.reshape(-1, 1, 2), H).reshape(-1, 2)
        intercept, slope, left_x, right_x = net_bottom
        between_posts = (net_pixels[:, 0] >= left_x) & (net_pixels[:, 0] <= right_x)
        if between_posts.sum() >= 5:
            gaps = np.abs(net_pixels[between_posts, 1] - (intercept + slope * net_pixels[between_posts, 0]))
            score = (score + 2 * float(np.mean(np.minimum(gaps, cap) ** 2))) / 3
        else:
            score += cap ** 2                               # the net line must be between the posts
    return score, on_line


def refine_court_points(frames, image_points, court_points, exclude_polygon=None, net_bottom=None, debug=False):
    """
    Start from the keypoint positions and move them until the drawn court lines sit on the real white lines
    (and the court's net line on the net bottom found in the image).
    Distances are capped, so a line hidden by a player or the lines of a neighbouring court can't pull it away.
    Many starting points are tried, because the keypoints can be far off.
    Returns better image points (4x2), or None if it didn't get better.
    """
    distance_map = white_line_distance_map(frames, exclude_polygon)
    samples = court_line_samples()
    court_points = np.asarray(court_points, np.float32)
    start = np.asarray(image_points, np.float64).reshape(-1)
    width = distance_map.shape[1]

    max_move = 0.10 * width                       # the model is off by some pixels, not by half a court

    def sensible(p):
        # points 8, 9 = far service line, 10, 11 = near service line (in the image)
        far_left, far_right, near_left, near_right = p.reshape(4, 2)
        far_above_near = far_left[1] < near_left[1] and far_right[1] < near_right[1]
        left_is_left = far_left[0] < far_right[0] and near_left[0] < near_right[0]
        far_looks_smaller = np.linalg.norm(far_right - far_left) < np.linalg.norm(near_right - near_left)
        return far_above_near and left_is_left and far_looks_smaller

    def score(p, cap):
        if not sensible(p):
            return cap ** 2 * 20                   # flipped / mirrored court: never accept
        value = fit_score(distance_map, p, court_points, samples, cap, net_bottom)[0]
        too_far = np.maximum(np.linalg.norm((p - start).reshape(4, 2), axis=1) - max_move, 0)
        return value + float(np.sum(too_far ** 2))

    def optimise(p, caps, max_evals):
        for cap in caps:
            p = minimize(lambda q: score(q, cap), p, method='Powell',
                         options={'xtol': 0.3, 'ftol': 1e-4, 'maxfev': max_evals}).x
        return p

    # 1. rough fit from many starts (the keypoints can be tens of pixels off)
    rng = np.random.default_rng(0)
    starts = [start] + [start + rng.normal(0, 0.03 * width, 8) for _ in range(30)]
    rough = sorted((optimise(s, (80, 40), 800) for s in starts), key=lambda p: score(p, 15))

    # 2. fine fit of the 3 best
    finals = [optimise(p, (15, 6), 3000) for p in rough[:3]]
    best = min(finals, key=lambda p: score(p, 6))

    _, on_line_before = fit_score(distance_map, start, court_points, samples, 6)
    _, on_line_after = fit_score(distance_map, best, court_points, samples, 6)
    if debug:
        print(f"    court lines on white pixels: {on_line_before:.0%} -> {on_line_after:.0%}")
    if on_line_after < max(on_line_before, 0.3):
        return None
    return best.reshape(4, 2).astype(np.float32)
