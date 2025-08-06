# grid_detector.py

import cv2
import numpy as np

# --- デフォルト定数 ---
CANNY_LOW        = 50
CANNY_HIGH       = 150
HOUGH_THRESH     = 80
MIN_LINE_LEN     = 100
MAX_LINE_GAP     = 10
VERT_THRESH_RATIO = 0.5
DEFAULT_GRID_COUNT = 10

def cluster_positions(positions, threshold):
    if not positions:
        return []
    clusters = []
    current = [positions[0]]
    for pos in positions[1:]:
        if abs(pos - current[-1]) <= threshold:
            current.append(pos)
        else:
            clusters.append(current)
            current = [pos]
    clusters.append(current)
    return [int(np.median(c)) for c in clusters]

def detect_board_grid(
    img,
    grid_count: int = DEFAULT_GRID_COUNT,
    canny_low: int = CANNY_LOW,
    canny_high: int = CANNY_HIGH,
    hough_thresh: int = HOUGH_THRESH,
    min_line_len: int = MIN_LINE_LEN,
    max_line_gap: int = MAX_LINE_GAP,
    vert_thresh_ratio: float = VERT_THRESH_RATIO
):
    """
    画像から GRID_COUNT x GRID_COUNT のグリッド交点リストを返す。
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(blur, canny_low, canny_high)

    h, w = gray.shape

    # 水平線検出
    lines = cv2.HoughLinesP(
        edges, 1, np.pi/180,
        threshold=hough_thresh,
        minLineLength=min_line_len,
        maxLineGap=max_line_gap
    )
    if lines is None:
        raise RuntimeError("水平線が検出できませんでした。パラメータを調整してください。")

    ys = [(y1+y2)//2 for x1,y1,x2,y2 in lines[:,0]
          if abs(y2 - y1) < abs(x2 - x1) * 0.2]
    cell_h = h / (grid_count - 1)
    y_centers = cluster_positions(sorted(ys), cell_h/2)

    if len(y_centers) < grid_count:
        y_grid = [int(i * h/(grid_count-1)) for i in range(grid_count)]
    else:
        y_centers.sort()
        y_grid = [y_centers[int(i*len(y_centers)/grid_count)] for i in range(grid_count)]

    # 垂直線検出
    hist_v = np.sum(edges == 255, axis=0)
    th_v = hist_v.max() * vert_thresh_ratio
    cols = np.where(hist_v > th_v)[0]
    cell_w = w / (grid_count - 1)
    x_centers = cluster_positions(sorted(cols), cell_w/2)

    if len(x_centers) < grid_count:
        x_grid = [int(i * w/(grid_count-1)) for i in range(grid_count)]
    else:
        x_centers.sort()
        x_grid = [x_centers[int(i*len(x_centers)/grid_count)] for i in range(grid_count)]

    # 交点配列を返す
    grid = np.zeros((grid_count, grid_count, 2), dtype=int)
    for i, y in enumerate(y_grid):
        for j, x in enumerate(x_grid):
            grid[i, j] = [x, y]
    return grid
