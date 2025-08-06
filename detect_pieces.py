#!/usr/bin/env python3
# detect_pieces_masked_vis_with_grid.py

import os
import glob
import cv2
import numpy as np
from grid_detector import detect_board_grid  # 汎用グリッド検出モジュールをインポート

# --- 設定 ---
SCREENSHOTS_DIR = "screenshots"
TEMPLATES_DIR   = "templates"
MATCH_THRESHOLD = 0.15   # 検出スコアの閾値
OUT_DIR         = "debug_vis_with_grid"
os.makedirs(OUT_DIR, exist_ok=True)

# --- テンプレート読み込み ---
templates = {}
for tpl_path in glob.glob(os.path.join(TEMPLATES_DIR, "*.png")):
    name = os.path.splitext(os.path.basename(tpl_path))[0]
    tpl = cv2.imread(tpl_path, cv2.IMREAD_UNCHANGED)
    if tpl is None or tpl.shape[2] < 4:
        print(f"⚠️ テンプレート読み込み失敗 or アルファなし: {tpl_path}")
        continue
    tpl_gray = cv2.cvtColor(tpl[:, :, :3], cv2.COLOR_BGR2GRAY)
    tpl_mask = tpl[:, :, 3]
    templates[name] = (tpl_gray, tpl_mask)
if not templates:
    raise RuntimeError("templates フォルダに透過PNGが見つかりません。")

# --- メイン処理 ---
for img_path in glob.glob(os.path.join(SCREENSHOTS_DIR, "*.png")):
    base = os.path.splitext(os.path.basename(img_path))[0]
    board_color = cv2.imread(img_path)
    if board_color is None:
        print(f"⚠️ 画像読み込み失敗: {img_path}")
        continue
    board_gray = cv2.cvtColor(board_color, cv2.COLOR_BGR2GRAY)

    # グリッド検出（grid_detectorモジュールを利用）
    grid = detect_board_grid(board_color, grid_count=10)
    print(f"Processing {base}, grid detected with {grid.shape[0]}x{grid.shape[1]} points.")

    # 各セルを走査し可視化
    rows, cols = grid.shape[0]-1, grid.shape[1]-1
    for r in range(rows):
        for c in range(cols):
            # グリッド交点からセル領域を得る
            x0, y0 = grid[r, c]
            x1, y1 = grid[r, c+1]
            x2, y2 = grid[r+1, c]
            cell_color = board_color[y0:y2, x0:x1]
            cell_gray  = board_gray[y0:y2, x0:x1]

            best_score = -1.0
            best_name  = None
            best_mask_r = None
            # 各テンプレートとのマッチング
            for name, (tpl_gray, tpl_mask) in templates.items():
                tpl_r  = cv2.resize(tpl_gray, (x1-x0, y2-y0))
                mask_r = cv2.resize(tpl_mask, (x1-x0, y2-y0))
                res    = cv2.matchTemplate(cell_gray, tpl_r, cv2.TM_CCORR_NORMED, mask=mask_r)
                _, score, _, _ = cv2.minMaxLoc(res)
                if score > best_score:
                    best_score, best_name, best_mask_r = score, name, mask_r.copy()

            # 空マスはオーバーレイ抑制
            if best_score > MATCH_THRESHOLD:
                mask_to_show = best_mask_r
            else:
                mask_to_show = np.zeros_like(best_mask_r)
                best_name = 'empty'

            # 可視化用画像作成
            mask_vis = cv2.cvtColor(mask_to_show, cv2.COLOR_GRAY2BGR)
            overlay = cell_color.copy()
            overlay[mask_to_show > 0] = (0,0,255)
            vis = np.hstack([cell_color, mask_vis, overlay])

            # 出力ファイル保存
            tag = f"{base}_r{r}c{c}_{best_name}_{best_score:.2f}.png"
            out_path = os.path.join(OUT_DIR, tag)
            cv2.imwrite(out_path, vis)

    print(f"→ saved visualizations to {OUT_DIR}/")
