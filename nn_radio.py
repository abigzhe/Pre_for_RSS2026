import cv2
import numpy as np
from pathlib import Path


# ============ 固定路径 ============
pairs_file = Path("/data1/home/limingzhe/abigzhe_dinopro/result/randomall/pairs_eval.txt")
features_root = Path("/data1/home/limingzhe/abigzhe_dinopro/result/randomall/superpoint&sift/features")
images_root = Path("/data1/home/limingzhe/abigzhe_dinopro/result/randomall/images")
output_root = Path("/data1/home/limingzhe/abigzhe_dinopro/result/randomall/superpoint&sift/matching")

methods = ["sift", "superpoint"]


# ============ NN-Ratio 匹配 ============
def match_ratio(desc1, desc2, max_desc=50000, ratio_thresh=0.75):
    """Nearest Neighbor Ratio (Lowe’s test) matching"""
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

    # 转 float32，保证 OpenCV 能处理
    desc1 = np.ascontiguousarray(desc1, dtype=np.float32)
    desc2 = np.ascontiguousarray(desc2, dtype=np.float32)

    # ⚡ 封顶抽样
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

    # 1->2 最近邻 & 次近邻
    knn_matches = bf.knnMatch(desc1, desc2, k=2)

    good_matches = []
    for i, m in enumerate(knn_matches):
        if len(m) < 2:
            continue
        best, second = m
        if best.distance < ratio_thresh * second.distance:
            # 恢复原始索引
            best.queryIdx = idx1[best.queryIdx]
            best.trainIdx = idx2[best.trainIdx]
            good_matches.append(best)

    return sorted(good_matches, key=lambda x: x.distance)


# ============ 特征加载 ============
def load_features(npz_file):
    data = np.load(npz_file)
    kps = [cv2.KeyPoint(float(x), float(y), 1) for x, y in data["keypoints"]]
    desc = data["descriptors"]

    if desc is None or len(desc) == 0:
        return kps, None, data["keypoints"]

    desc = np.ascontiguousarray(desc, dtype=np.float32)
    return kps, desc, data["keypoints"]


# ============ 可视化 ============
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


# ============ 路径查找 ============
def find_feature_file(features_root, scene, filename, method):
    scene_path = normalize_scene(scene)
    target = filename.replace(".jpg", f".{method}.npz")
    search_dir = features_root / scene_path
    candidates = list(search_dir.rglob(target))
    if len(candidates) == 0:
        return None
    return candidates[0]


def find_image_file(images_root, scene, filename):
    scene_path = normalize_scene(scene)
    search_dir = images_root / scene_path
    candidates = list(search_dir.rglob(filename))
    if len(candidates) == 0:
        return None
    return candidates[0]


def normalize_scene(scene: str) -> str:
    parts = scene.split("-", 2)
    if len(parts) == 1:
        return scene
    elif len(parts) == 2:
        return parts[0] + "/" + parts[1]
    else:
        return parts[0] + "/" + parts[1] + "-" + parts[2]


# ============ 主流程 ============
def main():
    pairs = [line.strip().split(",") for line in open(pairs_file, "r").readlines()[1:]]
    print(f"找到 {len(pairs)} 对图像对")

    for method in methods:
        out_dir = output_root / f"{method}+nnratio"
        vis_dir = out_dir / "vis"
        npz_dir = out_dir / "npz"
        vis_dir.mkdir(parents=True, exist_ok=True)
        npz_dir.mkdir(parents=True, exist_ok=True)

        for scene, f1, f2 in pairs:
            scene_path = Path(scene.replace("-", "/"))
            scene_npz_dir = npz_dir / scene_path
            scene_vis_dir = vis_dir / scene_path
            scene_npz_dir.mkdir(parents=True, exist_ok=True)
            scene_vis_dir.mkdir(parents=True, exist_ok=True)
            out_npz = scene_npz_dir / f"{Path(f1).stem}_{Path(f2).stem}.npz"
            out_vis = scene_vis_dir / f"{Path(f1).stem}_{Path(f2).stem}.jpg"

            if out_npz.exists() and out_vis.exists():
                print(f"⏭️ 跳过 {scene} {f1} {f2}, 匹配已存在")
                continue

            feat1_file = find_feature_file(features_root, scene, f1, method)
            feat2_file = find_feature_file(features_root, scene, f2, method)
            img1_file = find_image_file(images_root, scene, f1)
            img2_file = find_image_file(images_root, scene, f2)

            if feat1_file is None or feat2_file is None:
                print(f"⚠️ 跳过 {scene} {f1} {f2}, 特征不存在")
                continue
            if img1_file is None or img2_file is None:
                print(f"⚠️ 跳过 {scene} {f1} {f2}, 图像不存在")
                continue

            kps1, desc1, pts1 = load_features(feat1_file)
            kps2, desc2, pts2 = load_features(feat2_file)

            matches = match_ratio(desc1, desc2)
            if len(matches) == 0:
                print(f"⚠️ 跳过 {scene} {f1}-{f2}, 无有效匹配")
                continue

            print(f"[{method}] {scene}, {f1}-{f2}: {len(matches)} matches")

            matched_pts1 = np.array([pts1[m.queryIdx] for m in matches], dtype=np.float32)
            matched_pts2 = np.array([pts2[m.trainIdx] for m in matches], dtype=np.float32)
            np.savez(out_npz, pts1=matched_pts1, pts2=matched_pts2)

            img1 = cv2.imread(str(img1_file))
            img2 = cv2.imread(str(img2_file))
            vis_img = draw_matches_thick(img1, kps1, img2, kps2, matches, max_display=10, thickness=10)
            cv2.imwrite(str(out_vis), vis_img)


if __name__ == "__main__":
    main()
