#!/usr/bin/env python3
# detect_pieces_threshold.py

import os, glob
import cv2
import numpy as np

# --- 設定 ---
SCREENSHOTS_DIR  = "screenshots"
TEMPLATES_DIR    = "templates"
MATCH_THRESHOLD  = 0.15   # 先ほどのデバッグ結果を見ると max≈0.19 だったので、0.15 くらいから試す

# --- テンプレート読み込み ---
templates = {}
for tpl_path in glob.glob(os.path.join(TEMPLATES_DIR, "*.png")):
    name = os.path.splitext(os.path.basename(tpl_path))[0]
    tpl = cv2.imread(tpl_path, cv2.IMREAD_UNCHANGED)
    if tpl is None:
        print(f"⚠️ テンプレート読み込み失敗: {tpl_path}")
        continue
    # グレースケール化しておく
    tpl_gray = cv2.cvtColor(tpl[:, :, :3], cv2.COLOR_BGR2GRAY)
    templates[name] = tpl_gray

print("Loaded templates:", list(templates.keys()))
if not templates:
    raise RuntimeError("templates フォルダに PNG が見つかりません。")

# --- 画像ごとに処理 ---
for img_path in glob.glob(os.path.join(SCREENSHOTS_DIR, "*.png")):
    full = cv2.imread(img_path)
    if full is None:
        print(f"⚠️ 画像読み込み失敗: {img_path}")
        continue

    # 手動クロップ済みの盤面（450×450）を想定
    board = full
    board_gray = cv2.cvtColor(board, cv2.COLOR_BGR2GRAY)
    H, W = board_gray.shape
    cell_h, cell_w = H // 9, W // 9

    print(f"\n--- {os.path.basename(img_path)} size={W}×{H} cell={cell_w}×{cell_h} ---")
    # 全セルを走査
    for r in range(9):
        for c in range(9):
            y0, x0 = r*cell_h, c*cell_w
            cell = board_gray[y0:y0+cell_h, x0:x0+cell_w]

            best_score = -1.0
            best_piece = None
            # 各テンプレートとマッチング
            for name, tpl_gray in templates.items():
                tpl_r = cv2.resize(tpl_gray, (cell_w, cell_h))
                # 標準の COEFF_NORMED
                res = cv2.matchTemplate(cell, tpl_r, cv2.TM_CCOEFF_NORMED)
                _, score, _, _ = cv2.minMaxLoc(res)
                if score > best_score:
                    best_score, best_piece = score, name

            # 結果出力
            mark = "⚑ 検出!" if best_score > MATCH_THRESHOLD else ""
            print(f"Cell({r},{c}) → {best_piece} (score={best_score:.3f}) {mark}")

    print("--- end ---")
