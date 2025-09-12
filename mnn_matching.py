#!/usr/bin/env python3
import cv2
import numpy as np
from pathlib import Path
import random

# ============ 配置部分 ============
pairs_file = Path("/data1/home/limingzhe/abigzhe_dinopro/navi_v1.0/pairs.txt")
features_root = Path("/data1/home/limingzhe/abigzhe_dinopro/result/all/superpoint/pointmask_cpu")
images_root = Path("/data1/home/limingzhe/abigzhe_dinopro/navi_v1.0")
output_root = Path("/data1/home/limingzhe/abigzhe_dinopro/result/all/superpoint/match")

methods = ["sift", "superpoint"]

MAX_VISUALIZATIONS = 20  # 只画 20 张图


def match_mnn(desc1, desc2, max_desc=50000):
    if desc1 is None or desc2 is None:
        return []
    if not isinstance(desc1, np.ndarray) or not isinstance(desc2, np.ndarray):
        return []
    if desc1.ndim != 2 or desc2.ndim != 2:
        return []
    if desc1.shape[0] == 0 or desc2.shape[0] == 0:
        return []
    if desc1.shape[1] != desc2.shape[1]:
        print(f"⚠️ 描述子维度不一致: {desc1.shape} vs {desc2.shape}")
        return []

    desc1 = np.ascontiguousarray(desc1, dtype=np.float32)
    desc2 = np.ascontiguousarray(desc2, dtype=np.float32)

    if len(desc1) > max_desc:
        idx1 = np.random.choice(len(desc1), max_desc, replace=False)
        desc1 = desc1[idx1]
    else:
        idx1 = np.arange(len(desc1))

    if len(desc2) > max_desc:
        idx2 = np.random.choice(len(desc2), max_desc, replace=False)
        desc2 = desc2[idx2]
    else:
        idx2 = np.arange(len(desc2))

    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

    matches_12 = bf.knnMatch(desc1, desc2, k=1)
    nn12 = {i: m[0] for i, m in enumerate(matches_12) if len(m) > 0}

    matches_21 = bf.knnMatch(desc2, desc1, k=1)
    nn21 = {i: m[0] for i, m in enumerate(matches_21) if len(m) > 0}

    mutual_matches = []
    for i, m in nn12.items():
        j = m.trainIdx
        if j in nn21 and nn21[j].trainIdx == i:
            m.queryIdx = idx1[i]
            m.trainIdx = idx2[j]
            mutual_matches.append(m)

    return sorted(mutual_matches, key=lambda x: x.distance)


def load_features(npz_file):
    data = np.load(npz_file)
    kps = [cv2.KeyPoint(float(x), float(y), 1) for x, y in data["keypoints"]]
    desc = data["descriptors"]

    if desc is None or len(desc) == 0:
        return kps, None, data["keypoints"]

    desc = np.ascontiguousarray(desc, dtype=np.float32)
    return kps, desc, data["keypoints"]


def draw_matches_thick(img1, kps1, img2, kps2, matches, max_display=100, thickness=2):
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    out = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
    out[:h1, :w1] = img1
    out[:h2, w1:w1 + w2] = img2

    rng = np.random.default_rng(42)
    for m in matches[:max_display]:
        pt1 = tuple(map(int, kps1[m.queryIdx].pt))
        pt2 = tuple(map(int, kps2[m.trainIdx].pt))
        pt2 = (pt2[0] + w1, pt2[1])
        color = tuple(int(x) for x in rng.integers(0, 255, 3))
        cv2.line(out, pt1, pt2, color, thickness=thickness)
        cv2.circle(out, pt1, 3, color, -1)
        cv2.circle(out, pt2, 3, color, -1)
    return out


def find_feature_file(features_root, image_path, method):
    rel_path = Path(image_path).relative_to(images_root)
    target = rel_path.with_suffix(f".{method}.npz").name
    search_dir = features_root / rel_path.parent
    candidates = list(search_dir.rglob(target))
    if len(candidates) == 0:
        return None
    return candidates[0]


def main():
    all_pairs = [line.strip().split() for line in open(pairs_file, "r") if line.strip()]
    print(f"总共 {len(all_pairs)} 对图像对")

    # 前 MAX_VISUALIZATIONS 对用于可视化
    vis_indices = set(range(min(MAX_VISUALIZATIONS, len(all_pairs))))

    for method in methods:
        # npz 根目录（保持原始层级）
        npz_root = output_root / f"{method}+mnn_npz"
        vis_root = output_root / f"{method}+mnn_vis20"

        for idx, (img1_path, img2_path) in enumerate(all_pairs):
            feat1_file = find_feature_file(features_root, img1_path, method)
            feat2_file = find_feature_file(features_root, img2_path, method)

            if feat1_file is None or feat2_file is None:
                print(f"⚠️ 特征不存在，跳过: {img1_path}, {img2_path}")
                continue

            kps1, desc1, pts1 = load_features(feat1_file)
            kps2, desc2, pts2 = load_features(feat2_file)

            matches = match_mnn(desc1, desc2)
            if len(matches) == 0:
                print(f"⚠️ 无有效匹配，跳过: {img1_path}, {img2_path}")
                continue

            print(f"[{method}] {idx+1}/{len(all_pairs)} 匹配数: {len(matches)}")

            # 保存 npz（全部保存）
            rel_dir = Path(img1_path).relative_to(images_root).parent
            npz_dir = npz_root / rel_dir
            npz_dir.mkdir(parents=True, exist_ok=True)
            out_npz = npz_dir / f"{Path(img1_path).stem}_{Path(img2_path).stem}.npz"

            matched_pts1 = np.array([pts1[m.queryIdx] for m in matches], dtype=np.float32)
            matched_pts2 = np.array([pts2[m.trainIdx] for m in matches], dtype=np.float32)
            np.savez(out_npz, pts1=matched_pts1, pts2=matched_pts2)

            # 只前 MAX_VISUALIZATIONS 对画可视化
            if idx in vis_indices:
                vis_dir = vis_root / rel_dir
                vis_dir.mkdir(parents=True, exist_ok=True)

                img1 = cv2.imread(img1_path)
                img2 = cv2.imread(img2_path)
                vis_img = draw_matches_thick(img1, kps1, img2, kps2,
                                             matches, max_display=10, thickness=10)
                out_vis = vis_dir / f"{Path(img1_path).stem}_{Path(img2_path).stem}.jpg"
                cv2.imwrite(str(out_vis), vis_img)


if __name__ == "__main__":
    main()
