#!/usr/bin/env python3
# detect_board_grid.py

import os
import glob
import cv2
from grid_detector import detect_board_grid  # ← ここでインポート

# --- 設定 ---
SCREENSHOTS_DIR = "screenshots"
OUT_DIR         = "grid_vis_hybrid"
GRID_COUNT      = 10  # 必要に応じて上書き可能

os.makedirs(OUT_DIR, exist_ok=True)

if __name__ == '__main__':
    for img_path in glob.glob(os.path.join(SCREENSHOTS_DIR, "*.png")):
        img = cv2.imread(img_path)
        if img is None:
            print(f"⚠️ 画像読み込み失敗: {img_path}")
            continue
        try:
            grid = detect_board_grid(img, grid_count=GRID_COUNT)
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
