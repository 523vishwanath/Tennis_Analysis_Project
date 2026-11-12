import cv2
import numpy as np


def make_background(frames):
    # median of several frames: players / ball walking in front of the net disappear
    if len(frames) == 1:
        return frames[0]
    return np.median(np.stack(frames), axis=0).astype(np.uint8)


def fit_net_tape(points, center_x):
    # tape = 2 straight lines that meet in the middle (it sags at the center strap)
    # y = c + m_left * (x - center_x) on the left, c + m_right * (x - center_x) on the right
    x, y = points[:, 0], points[:, 1]
    dx = x - center_x
    A = np.column_stack([np.ones_like(x), np.where(dx < 0, dx, 0), np.where(dx >= 0, dx, 0)])
    coeffs, *_ = np.linalg.lstsq(A, y, rcond=None)
    return coeffs


def tape_y(coeffs, x, center_x):
    dx = np.asarray(x, float) - center_x
    return coeffs[0] + np.where(dx < 0, dx * coeffs[1], 0) + np.where(dx >= 0, dx * coeffs[2], 0)


def detect_net_tape(frames, base_points, prior_tape_points, court_xs, alleys, debug=False, check_alleys=True):
    """
    Finds the white net tape in the image.
    base_points: bottom of the net (from the homography, reliable at any angle), post to post
    prior_tape_points: rough guess of the tape (from geometry), only used to know where to look
    court_xs: court x (meters) of every base point
    alleys: [(x_from, x_to), (x_from, x_to)] in meters, the 2 doubles alleys (the tape always crosses them,
            a service line never does)
    Returns {'coeffs', 'center_x', 'left_x', 'right_x'} (tape shape + where it starts/ends = the posts),
    or None if the tape can't be found reliably.
    """
    image = make_background(frames)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    height_img, width_img = gray.shape

    # thin bright lines stand out with a top-hat (bright thing smaller than the kernel, minus background)
    prior_heights = np.linalg.norm(base_points - prior_tape_points, axis=1)
    kernel_size = int(max(9, np.median(prior_heights) * 0.4)) | 1
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_size)))

    left_x = min(base_points[0, 0], prior_tape_points[0, 0])
    right_x = max(base_points[-1, 0], prior_tape_points[-1, 0])
    center_x = (base_points[len(base_points) // 2, 0] + prior_tape_points[len(prior_tape_points) // 2, 0]) / 2
    order = np.argsort(base_points[:, 0])
    # search 25% past where the keypoints put the posts: the keypoints can be off, the real tape ends must be inside
    span = right_x - left_x
    columns = np.arange(int(left_x - 0.25 * span), int(right_x + 0.25 * span), 2)
    columns = columns[(columns >= 0) & (columns < width_img)]      # part of the net can be outside the frame
    if len(columns) < 20:
        return None

    candidates = []
    examined = set()                          # columns that were looked at (not skipped as a crossing line)
    for x in columns:
        base_y = np.interp(x, base_points[order, 0], base_points[order, 1])          # flat beyond the ends
        h = np.interp(x, base_points[order, 0], prior_heights[order])
        y_top, y_bottom = int(max(0, base_y - 1.8 * h)), int(min(height_img - 1, base_y - 0.35 * h))
        if y_bottom - y_top < 5:
            continue
        column = tophat[y_top:y_bottom, int(x)]
        threshold = max(12.0, 0.4 * column.max())
        bright = column > threshold
        if bright.mean() > 0.35:              # a court line crossing the band (sideline / center line): unknown, skip
            continue
        examined.add(x)                       # looked here: either tape or NO tape
        if column.max() < 12:                 # nothing bright here
            continue
        # every bright peak in this column is a candidate (tape AND e.g. a service line), RANSAC picks the tape
        peaks = [k for k in range(1, len(column) - 1)
                 if column[k] > threshold and column[k] >= column[k - 1] and column[k] >= column[k + 1]]
        for k in sorted(peaks, key=lambda k: -column[k])[:4]:
            candidates.append((x, y_top + k))

    if len(candidates) < 20:
        return None
    points = np.array(candidates, float)

    # the tape runs post to post: it must also be found in BOTH doubles alleys (outside the singles lines),
    # where a service line never is
    def image_x(court_x):
        return np.interp(court_x, court_xs, base_points[:, 0])
    (a1, a2), (b1, b2) = alleys
    lo1, hi1 = sorted((image_x(a1), image_x(a2)))
    lo2, hi2 = sorted((image_x(b1), image_x(b2)))
    outer_left = columns[(columns >= lo1) & (columns <= hi1)]
    outer_right = columns[(columns >= lo2) & (columns <= hi2)]

    def coverage_of(inlier_points, cols):
        if len(cols) == 0:
            return 0.0
        return len(np.intersect1d(inlier_points[:, 0], cols)) / len(cols)

    # RANSAC: try many 3-point fits, keep the one most COLUMNS agree with
    tolerance = max(2.0, 0.05 * np.median(prior_heights))
    ends = np.r_[outer_left, outer_right]

    def ransac(center, tries=600, seed=0):
        rng = np.random.default_rng(seed)
        best_inliers, best_score = None, -1
        for _ in range(tries):
            sample = points[rng.choice(len(points), 3, replace=False)]
            if (sample[:, 0] < center).all() or (sample[:, 0] >= center).all():
                continue                       # need points on both sides to fit both halves
            coeffs = fit_net_tape(sample, center)
            inliers = np.abs(tape_y(coeffs, points[:, 0], center) - points[:, 1]) < tolerance
            inlier_points = points[inliers]
            score = len(np.unique(inlier_points[:, 0]))                       # columns, not peaks
            score += 2 * len(np.unique(inlier_points[:, 0][np.isin(inlier_points[:, 0], ends)]))   # ends count extra
            if score > best_score:
                best_inliers, best_score = inliers, score
        return best_inliers, best_score

    best_inliers, _ = ransac(center_x)
    if best_inliers is None:
        return None

    # the bend (center strap) may not be where the keypoints say: try positions along the found tape
    xs_found = points[best_inliers, 0]
    lo, hi = np.percentile(xs_found, 2), np.percentile(xs_found, 98)
    best_center, best_score = center_x, -1
    for candidate in np.linspace(lo + 0.25 * (hi - lo), lo + 0.75 * (hi - lo), 21):
        inliers, score = ransac(candidate, tries=250, seed=1)
        if inliers is not None and score > best_score:
            best_center, best_score, best_inliers = candidate, score, inliers
    center_x = best_center

    inlier_points = points[best_inliers]
    coeffs = fit_net_tape(inlier_points, center_x)

    # must be a real tape: found along most of the net, and in every doubles alley that is in the frame
    coverage = len(np.unique(inlier_points[:, 0])) / max(len(columns), 1)
    visible_alleys = [cols for cols in (outer_left, outer_right) if len(cols) >= 5]
    alley_covers = [coverage_of(inlier_points, cols) for cols in visible_alleys]
    left_cover = coverage_of(inlier_points, outer_left)
    right_cover = coverage_of(inlier_points, outer_right)

    tape_points = np.column_stack([prior_tape_points[:, 0], tape_y(coeffs, prior_tape_points[:, 0], center_x)])

    # its height must be believable compared to the rough guess
    found_heights = np.linalg.norm(base_points - tape_points, axis=1)
    height_ratio = np.median(found_heights / np.maximum(prior_heights, 1))

    alleys_ok = (len(visible_alleys) >= 1 and min(alley_covers) >= 0.25) if check_alleys else True
    ok = coverage >= 0.35 and alleys_ok and 0.6 <= height_ratio <= 1.6
    if debug:
        print(f"    tape search: {len(points)} candidates, coverage {coverage:.2f} (alleys L {left_cover:.2f} R {right_cover:.2f}), height ratio {height_ratio:.2f} -> {'found' if ok else 'not found'}")
    if not ok:
        return None
    # the tape ends at the posts. The real tape is continuous, with only short gaps (a court line crossing it,
    # a logo, the center strap). Walk out from the middle; stop at a long EMPTY gap (no tape where we looked).
    tape_columns = set(np.unique(inlier_points[:, 0]).tolist())
    core = np.percentile(inlier_points[:, 0], [10, 90])
    max_gap = 0.06 * (core[1] - core[0]) * 1.25
    middle = int(np.argmin(np.abs(columns - center_x)))

    max_crossing = 0.03 * (core[1] - core[0]) * 1.25      # a court line crossing the tape is narrow

    def walk(direction):
        last_tape, gap, unknown_run = columns[middle], 0.0, 0.0
        index = middle
        while 0 <= index + direction < len(columns):
            index += direction
            x = columns[index]
            if x in tape_columns:
                last_tape, gap, unknown_run = x, 0.0, 0.0
                continue
            if x in examined:
                gap += 2                       # looked here, no tape (columns are 2 px apart)
                unknown_run = 0.0
            else:
                unknown_run += 2               # skipped as a bright crossing line...
                if unknown_run > max_crossing: # ...but too wide for a line (a door, a banner, netting) -> no tape
                    gap += 2
            if gap > max_gap:
                break
        return last_tape

    left_end, right_end = walk(-1), walk(+1)
    inlier_xs = np.array([left_end, right_end], float)
    return {'coeffs': coeffs, 'center_x': center_x, 'left_x': float(inlier_xs[0]), 'right_x': float(inlier_xs[-1])}



def detect_net_bottom(frames, tape, rough_height_px):
    """
    Finds the bottom of the net under an already found tape.
    Two clues, either one is enough:
      - the net MESH is a strongly textured band, the court below it is smooth -> texture stops
      - the net's shadow / bottom band is darker than the court -> a dark-to-bright step going down
    Each column picks its best row, then one straight line is fitted through them (RANSAC).
    Returns (intercept, slope) of the bottom line y = intercept + slope * x, or None.
    """
    image = make_background(frames)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    height_img, width_img = gray.shape
    texture = cv2.blur(np.abs(cv2.Laplacian(gray, cv2.CV_32F, ksize=3)), (9, 3))
    smooth = cv2.blur(gray, (9, 1))

    left_x, right_x = tape['left_x'], tape['right_x']
    columns = np.arange(int(left_x) + 4, int(right_x) - 4, 2)
    columns = columns[(columns >= 0) & (columns < width_img)]
    if len(columns) < 20:
        return None

    step = max(3, int(rough_height_px * 0.06))
    points = []
    for x in columns:
        top = float(tape_y(tape['coeffs'], x, tape['center_x']))
        y_from = int(top + 0.45 * rough_height_px)
        y_to = int(min(height_img - step - 1, top + 2.0 * rough_height_px))
        if y_to - y_from < 4:
            continue
        ys = np.arange(y_from, y_to)
        tex_col, gray_col = texture[:, int(x)], smooth[:, int(x)]
        mesh_level = np.median(tex_col[int(top) + 3:int(top + 0.4 * rough_height_px) + 4]) + 1e-3
        best_y, best_score = None, 0.0
        for y in ys:
            texture_drop = (tex_col[y - step:y].mean() - tex_col[y:y + step].mean()) / mesh_level
            brightness_step = (gray_col[y:y + step].mean() - gray_col[y - step:y].mean()) / 15.0
            score = max(texture_drop, 0) + max(brightness_step, 0)
            if score > best_score:
                best_y, best_score = y, score
        if best_y is not None and best_score > 0.5:
            points.append((x, best_y))

    if len(points) < 0.3 * len(columns):
        return None
    points = np.array(points, float)

    rng = np.random.default_rng(0)
    tolerance = max(2.0, 0.05 * rough_height_px)
    best_inliers = None
    for _ in range(300):
        i, j = rng.choice(len(points), 2, replace=False)
        if points[i, 0] == points[j, 0]:
            continue
        slope = (points[j, 1] - points[i, 1]) / (points[j, 0] - points[i, 0])
        intercept = points[i, 1] - slope * points[i, 0]
        inliers = np.abs(intercept + slope * points[:, 0] - points[:, 1]) < tolerance
        if best_inliers is None or inliers.sum() > best_inliers.sum():
            best_inliers = inliers
    if best_inliers is None or best_inliers.sum() < 0.3 * len(columns):
        return None

    slope, intercept = np.polyfit(points[best_inliers, 0], points[best_inliers, 1], 1)
    return float(intercept), float(slope)



def find_net_tape(frames, base_points, prior_tape_points, court_xs, alleys, debug=False):
    """
    Pass 1 searches where the court keypoints say the net is (they can be off).
    Then it searches again around the tape it found, a bit wider each time, until the tape ends stop moving.
    So the result doesn't depend on how good the keypoints are.
    """
    found = detect_net_tape(frames, base_points, prior_tape_points, court_xs, alleys, debug=debug)
    if found is None:
        return None

    rough_height = float(np.median(np.linalg.norm(base_points - prior_tape_points, axis=1)))
    for _ in range(4):
        span = found['right_x'] - found['left_x']
        xs = np.linspace(found['left_x'] - 0.35 * span, found['right_x'] + 0.35 * span, 21)
        tape_guess = np.column_stack([xs, tape_y(found['coeffs'], xs, found['center_x'])])
        again = detect_net_tape(frames, tape_guess + [0, rough_height], tape_guess, np.linspace(0, 1, 21), alleys,
                                debug=debug, check_alleys=False)
        if again is None:
            break
        moved = abs(again['left_x'] - found['left_x']) + abs(again['right_x'] - found['right_x'])
        found = again
        if moved < 10:
            break
    return found


def refine_posts_with_mesh(frames, tape, rough_height_px, debug=False):
    """
    The net mesh only exists BETWEEN the posts. Under the tape, mesh = strong fine texture.
    Pull each post inward until there is mesh under the tape. If the mesh is too fine to see
    in this video (low resolution), keep the tape ends as they are.
    """
    image = make_background(frames)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    height_img, width_img = gray.shape
    texture = cv2.blur(np.abs(cv2.Laplacian(gray, cv2.CV_32F, ksize=3)), (3, 3))

    columns = np.arange(int(tape['left_x']), int(tape['right_x']) + 1, 2)
    columns = columns[(columns >= 0) & (columns < width_img)]
    if len(columns) < 20:
        return tape

    mesh = []
    for x in columns:
        top = float(tape_y(tape['coeffs'], x, tape['center_x']))
        y_from, y_to = int(top + 0.2 * rough_height_px), int(top + 0.7 * rough_height_px)
        y_from, y_to = max(0, y_from), min(height_img - 1, max(y_to, y_from + 2))
        mesh.append(np.median(texture[y_from:y_to, int(x)]))
    mesh = np.array(mesh)

    # is the mesh visible at all? compare the middle of the net with the court far below it
    middle_columns = slice(int(len(columns) * 0.3), int(len(columns) * 0.7))
    mesh_level = np.median(mesh[middle_columns])
    court_rows = []
    for x in columns[middle_columns]:
        top = float(tape_y(tape['coeffs'], x, tape['center_x']))
        y_from = int(min(height_img - 3, top + 1.8 * rough_height_px))
        court_rows.append(np.median(texture[y_from:min(height_img, y_from + int(0.4 * rough_height_px) + 2), int(x)]))
    court_level = np.median(court_rows) if court_rows else 0
    if mesh_level < 2 * court_level + 3:
        if debug:
            print("    posts: mesh not visible in this video, keeping the tape ends")
        return tape

    has_mesh = mesh > 0.45 * mesh_level
    max_gap = 0.05 * (columns[-1] - columns[0])
    middle = len(columns) // 2

    def walk(direction):
        last, gap, index = columns[middle], 0, middle
        while 0 <= index + direction < len(columns):
            index += direction
            if has_mesh[index]:
                last, gap = columns[index], 0
            else:
                gap += 2
                if gap > max_gap:
                    break
        return float(last)

    refined = dict(tape)
    refined['left_x'], refined['right_x'] = walk(-1), walk(+1)
    if debug:
        print(f"    posts: tape ends {tape['left_x']:.0f}-{tape['right_x']:.0f} -> mesh ends {refined['left_x']:.0f}-{refined['right_x']:.0f}")
    return refined



def snap_posts(frames, tape, bottom, debug=False):
    """
    A post is a dark pole from the tape down to the bottom of the net. Near each estimated end,
    pick the column that is darkest over that whole height. Keep the estimate if no pole stands out.
    """
    image = make_background(frames)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    height_img, width_img = gray.shape
    intercept, slope = bottom
    span = tape['right_x'] - tape['left_x']
    search = 0.07 * span

    def band_brightness(x):
        top = float(tape_y(tape['coeffs'], x, tape['center_x']))
        low = intercept + slope * x
        y0, y1 = int(max(0, top)), int(min(height_img - 1, low))
        if y1 - y0 < 4:
            return None
        return float(np.mean(gray[y0:y1, int(x)]))

    def snap(estimate):
        xs = np.arange(int(estimate - search), int(estimate + search) + 1)
        xs = xs[(xs >= 0) & (xs < width_img)]
        values = [(x, band_brightness(x)) for x in xs]
        values = [(x, v) for x, v in values if v is not None]
        if len(values) < 5:
            return estimate
        brightness = np.array([v for _, v in values])
        darkest = int(np.argmin(brightness))
        # the pole must clearly stand out from the rest of the search area
        if np.median(brightness) - brightness[darkest] < 25:
            return estimate
        return float(values[darkest][0])

    snapped = dict(tape)
    snapped['left_x'], snapped['right_x'] = snap(tape['left_x']), snap(tape['right_x'])
    if debug:
        print(f"    posts snapped to poles: {tape['left_x']:.0f}-{tape['right_x']:.0f} -> {snapped['left_x']:.0f}-{snapped['right_x']:.0f}")
    return snapped
