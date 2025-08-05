#!/usr/bin/env python3
# detect_grid_hybrid_vis.py

import os
import glob
import cv2
import numpy as np

# --- 設定 ---
SCREENSHOTS_DIR = "screenshots"
OUT_DIR         = "grid_vis_hybrid"
# エッジ検出閾値
CANNY_LOW       = 50
CANNY_HIGH      = 150
# HoughLinesP（水平線用）のパラメータ
HOUGH_THRESH    = 80
MIN_LINE_LEN    = 100
MAX_LINE_GAP    = 10
# グリッド数
GRID_COUNT      = 10
# 垂直投影の閾値割合（縦ラインの強度を決める）
VERT_THRESH_RATIO = 0.5

os.makedirs(OUT_DIR, exist_ok=True)

# --- クラスタリング関数 ---
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
    centers = [int(np.median(c)) for c in clusters]
    return centers

# --- グリッド検出: 水平はHoughLinesP, 垂直は投影プロファイル ---
def detect_board_grid(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    edges= cv2.Canny(blur, CANNY_LOW, CANNY_HIGH)

    h, w = gray.shape
    # 水平線検出 (HoughLinesP)
    lines = cv2.HoughLinesP(
        edges, 1, np.pi/180,
        threshold=HOUGH_THRESH,
        minLineLength=MIN_LINE_LEN,
        maxLineGap=MAX_LINE_GAP
    )
    if lines is None:
        raise RuntimeError("水平線が検出できませんでした。パラメータを調整してください。")
    ys = []
    for x1,y1,x2,y2 in lines[:,0]:
        if abs(y2 - y1) < abs(x2 - x1) * 0.2:
            ys.append((y1+y2)//2)
    # クラスタリング
    cell_h = h / (GRID_COUNT - 1)
    y_centers = cluster_positions(sorted(ys), cell_h/2)
    # 必要な本数に補正
    if len(y_centers) < GRID_COUNT:
        start, end = 0, h
        y_grid = [int(start + (end - start)*i/(GRID_COUNT-1)) for i in range(GRID_COUNT)]
    else:
        y_centers = sorted(y_centers)
        y_grid = [y_centers[int(i*len(y_centers)/GRID_COUNT)] for i in range(GRID_COUNT)]

    # 垂直線検出 (投影プロファイル)
    hist_v = np.sum(edges == 255, axis=0)
    th_v = hist_v.max() * VERT_THRESH_RATIO
    cols = np.where(hist_v > th_v)[0]
    # クラスタリング
    cell_w = w / (GRID_COUNT - 1)
    x_centers = cluster_positions(sorted(cols), cell_w/2)
    # 補正
    if len(x_centers) < GRID_COUNT:
        start, end = 0, w
        x_grid = [int(start + (end - start)*i/(GRID_COUNT-1)) for i in range(GRID_COUNT)]
    else:
        x_centers = sorted(x_centers)
        x_grid = [x_centers[int(i*len(x_centers)/GRID_COUNT)] for i in range(GRID_COUNT)]

    # 交点配列
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
        for y in grid[:,0,1]: cv2.line(vis, (0,y), (w,y), (0,255,0), 2)
        # 垂直線
        for x in grid[0,:,0]: cv2.line(vis, (x,0), (x,h), (0,255,0), 2)
        # 交点
        for i in range(GRID_COUNT):
            for j in range(GRID_COUNT):
                x,y = grid[i,j]
                cv2.circle(vis, (x,y), 5, (0,0,255), -1)

        base = os.path.splitext(os.path.basename(img_path))[0]
        out = os.path.join(OUT_DIR, f"{base}_hybrid_grid.png")
        cv2.imwrite(out, vis)
        print(f"Saved hybrid grid visualization: {out}")
