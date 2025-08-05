#!/usr/bin/env python3
# detect_grid_houghP_vis.py

import os
import glob
import cv2
import numpy as np

# --- 設定 ---
SCREENSHOTS_DIR = "screenshots"
OUT_DIR         = "grid_vis_houghP"
# Canny エッジ検出
CANNY_LOW       = 50
CANNY_HIGH      = 150
# HoughLinesP のパラメータ
HOUGH_THRESH    = 80
MIN_LINE_LEN    = 100
MAX_LINE_GAP    = 10
# グリッド線数
GRID_COUNT      = 10
os.makedirs(OUT_DIR, exist_ok=True)

# --- クラスタリングによる代表線の選出 ---
def cluster_positions(positions, threshold):
    """
    ソートされた位置リストをグループ化し、しきい値以内の連続位置を1つのクラスタにまとめ
    各クラスタの中央値を返す
    """
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
    centers = [int(np.median(c)) for c in clusters]
    return centers

# --- グリッド検出関数 (HoughLinesP + クラスタリング) ---
def detect_board_grid(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    edges= cv2.Canny(blur, CANNY_LOW, CANNY_HIGH)

    h, w = gray.shape
    cell_h = h / (GRID_COUNT-1)
    cell_w = w / (GRID_COUNT-1)
    thr_y = cell_h / 2
    thr_x = cell_w / 2

    # HoughLinesP: 短い線は除去
    lines = cv2.HoughLinesP(edges, 1, np.pi/180,
                             threshold=HOUGH_THRESH,
                             minLineLength=MIN_LINE_LEN,
                             maxLineGap=MAX_LINE_GAP)
    if lines is None:
        raise RuntimeError("ライン検出失敗: HoughLinesPのパラメータを調整してください。")

    ys = []
    xs = []
    # 水平線・垂直線に振り分け
    for x1,y1,x2,y2 in lines[:,0]:
        dx = x2 - x1
        dy = y2 - y1
        if abs(dy) < abs(dx) * 0.2:  # 水平ほぼ
            ys.append((y1+y2)//2)
        elif abs(dx) < abs(dy) * 0.2:  # 垂直ほぼ
            xs.append((x1+x2)//2)

    # クラスタリング
    y_centers = cluster_positions(sorted(ys), thr_y)
    x_centers = cluster_positions(sorted(xs), thr_x)

    # 必要な本数になるよう補正
    def adjust(centers, limit):
        if len(centers) >= limit:
            # 均等に間引き
            return [centers[int(i*len(centers)/limit)] for i in range(limit)]
        else:
            # 等間隔補完
            start, end = 0, (h if limit==GRID_COUNT else w)
            return [int(start + (end-start)*i/(limit-1)) for i in range(limit)]

    y_grid = adjust(y_centers, GRID_COUNT)
    x_grid = adjust(x_centers, GRID_COUNT)

    # 交点計算
    grid = np.zeros((GRID_COUNT, GRID_COUNT, 2), dtype=int)
    for i, y in enumerate(y_grid):
        for j, x in enumerate(x_grid):
            grid[i,j] = [x, y]
    return grid

# --- 可視化 ---
if __name__ == '__main__':
    for img_path in glob.glob(os.path.join(SCREENSHOTS_DIR, "*.png")):
        img = cv2.imread(img_path)
        if img is None:
            print(f"⚠️ 画像読み込み失敗: {img_path}")
            continue

        try:
            grid = detect_board_grid(img)
        except Exception as e:
            print(f"グリッド検出エラー ({img_path}): {e}")
            continue

        vis = img.copy()
        h, w = vis.shape[:2]

        # 水平線
        for y in grid[:,0,1]:
            cv2.line(vis, (0,y), (w,y), (0,255,0), 2)
        # 垂直線
        for x in grid[0,:,0]:
            cv2.line(vis, (x,0), (x,h), (0,255,0), 2)
        # 交点
        for i in range(GRID_COUNT):
            for j in range(GRID_COUNT):
                x, y = grid[i,j]
                cv2.circle(vis, (x,y), 5, (0,0,255), -1)

        base = os.path.splitext(os.path.basename(img_path))[0]
        out = os.path.join(OUT_DIR, f"{base}_houghP_grid.png")
        cv2.imwrite(out, vis)
        print(f"Saved grid visualization: {out}")