#!/usr/bin/env python3
import cv2
import numpy as np
from pathlib import Path

# ============ 配置 ============
# 每行两个绝对路径：/path/to/imgA.jpg /path/to/imgB.jpg
pairs_file = Path("/data1/home/limingzhe/abigzhe_dinopro/navi_v1.0/pairs.txt")
features_root = Path("/data1/home/limingzhe/abigzhe_dinopro/result/all/superpoint/pointmask_cpu")
images_root = Path("/data1/home/limingzhe/abigzhe_dinopro/navi_v1.0")
output_root = Path("/data1/home/limingzhe/abigzhe_dinopro/result/all/superpoint/match")
methods = ["sift", "superpoint"]


MAX_VISUALIZATIONS = 20
MAX_DESC = 50000

# 不同方法对应的 ratio 阈值
ratio_thresh_map = {
    "sift": 0.8,
    "superpoint": 0.9
}


def make_rel_key(image_path: str | Path, anchors: list[Path]) -> Path:
    p = Path(image_path)
    for root in anchors:
        try:
            return p.relative_to(root)
        except Exception:
            pass
    if "navi_v1.0" in p.parts:
        idx = p.parts.index("navi_v1.0")
        return Path(*p.parts[idx+1:])
    return Path(*p.parts[-4:])


def find_feature_file(image_path: str | Path, method: str, images_root: Path, features_root: Path, dataset_root: Path) -> Path | None:
    rel_key = make_rel_key(image_path, anchors=[images_root, dataset_root])
    target_name = rel_key.with_suffix(f".{method}.npz").name
    search_dir = features_root / rel_key.parent
    candidates = list(search_dir.rglob(target_name))
    if len(candidates) == 0:
        return None
    return candidates[0]


def match_ratio(desc1, desc2, ratio_thresh=0.75, max_desc=MAX_DESC):
    if desc1 is None or desc2 is None: return []
    if not isinstance(desc1, np.ndarray) or not isinstance(desc2, np.ndarray): return []
    if desc1.ndim != 2 or desc2.ndim != 2: return []
    if desc1.shape[0] == 0 or desc2.shape[0] == 0: return []
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
    knn_matches = bf.knnMatch(desc1, desc2, k=2)

    good = []
    for pair in knn_matches:
        if len(pair) < 2:
            continue
        best, second = pair
        if best.distance < ratio_thresh * second.distance:
            best.queryIdx = int(idx1[best.queryIdx])
            best.trainIdx = int(idx2[best.trainIdx])
            good.append(best)

    return sorted(good, key=lambda x: x.distance)


def load_features(npz_file: Path):
    data = np.load(npz_file)
    kps = [cv2.KeyPoint(float(x), float(y), 1) for x, y in data["keypoints"]]
    desc = data["descriptors"]
    if desc is None or len(desc) == 0:
        return kps, None, data["keypoints"]
    desc = np.ascontiguousarray(desc, dtype=np.float32)
    return kps, desc, data["keypoints"]


def draw_matches_thick(img1, kps1, img2, kps2, matches, max_display=10, thickness=10):
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


def main():
    dataset_root = pairs_file.parent
    lines = [line.strip() for line in open(pairs_file, "r") if line.strip()]
    pairs = [ln.split() for ln in lines]
    print(f"读取到 {len(pairs)} 对图像对")

    vis_indices = set(range(min(MAX_VISUALIZATIONS, len(pairs))))

    for method in methods:
        npz_root = output_root / f"{method}+nnratio_npz"
        vis_root = output_root / f"{method}+nnratio_vis20"
        ratio_thresh = ratio_thresh_map.get(method, 0.75)

        for idx, (img1_path, img2_path) in enumerate(pairs):
            feat1_file = find_feature_file(img1_path, method, images_root, features_root, dataset_root)
            feat2_file = find_feature_file(img2_path, method, images_root, features_root, dataset_root)
            if feat1_file is None or feat2_file is None:
                print(f"⚠️ 特征缺失，跳过:\n  {img1_path}\n  {img2_path}")
                continue

            kps1, desc1, pts1 = load_features(feat1_file)
            kps2, desc2, pts2 = load_features(feat2_file)

            matches = match_ratio(desc1, desc2, ratio_thresh=ratio_thresh, max_desc=MAX_DESC)
            if len(matches) == 0:
                print(f"⚠️ 无有效匹配，跳过:\n  {img1_path}\n  {img2_path}")
                continue

            print(f"[{method}] {idx+1}/{len(pairs)} 匹配数: {len(matches)}")

            rel_dir = make_rel_key(img1_path, anchors=[images_root, dataset_root]).parent
            out_dir_npz = npz_root / rel_dir
            out_dir_npz.mkdir(parents=True, exist_ok=True)
            out_npz = out_dir_npz / f"{Path(img1_path).stem}_{Path(img2_path).stem}.npz"

            matched_pts1 = np.array([pts1[m.queryIdx] for m in matches], dtype=np.float32)
            matched_pts2 = np.array([pts2[m.trainIdx] for m in matches], dtype=np.float32)
            np.savez(out_npz, pts1=matched_pts1, pts2=matched_pts2)

            if idx in vis_indices:
                out_dir_vis = vis_root / rel_dir
                out_dir_vis.mkdir(parents=True, exist_ok=True)

                img1 = cv2.imread(img1_path)
                img2 = cv2.imread(img2_path)
                if img1 is None or img2 is None:
                    print(f"⚠️ 读取原图失败，跳过可视化:\n  {img1_path}\n  {img2_path}")
                else:
                    vis = draw_matches_thick(img1, kps1, img2, kps2, matches)
                    out_vis = out_dir_vis / f"{Path(img1_path).stem}_{Path(img2_path).stem}.jpg"
                    cv2.imwrite(str(out_vis), vis)


if __name__ == "__main__":
    main()
