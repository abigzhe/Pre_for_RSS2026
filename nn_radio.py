
import cv2
import numpy as np
from pathlib import Path


# ============ 固定路径 ============
pairs_file = Path("/data1/home/limingzhe/abigzhe_dinopro/result/randomall/pairs_eval.txt")
features_root = Path("/data1/home/limingzhe/abigzhe_dinopro/result/randomall/superpoint&sift/features")
images_root = Path("/data1/home/limingzhe/abigzhe_dinopro/result/randomall/images")
output_root = Path("/data1/home/limingzhe/abigzhe_dinopro/result/randomall/superpoint&sift/matching")

methods = ["sift", "superpoint"]


# ============ 函数部分 ============
def match_mnn(desc1, desc2):
    """Mutual Nearest Neighbor (MNN) matching with BFMatcher"""
    if desc1 is None or desc2 is None or len(desc1) == 0 or len(desc2) == 0:
        return []

    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

    # 1 -> 2
    matches_12 = bf.match(desc1, desc2)
    nn12 = {m.queryIdx: m for m in matches_12}

    # 2 -> 1
    matches_21 = bf.match(desc2, desc1)
    nn21 = {m.queryIdx: m for m in matches_21}

    # mutual check
    mutual_matches = []
    for i, m in nn12.items():
        j = m.trainIdx
        if j in nn21 and nn21[j].trainIdx == i:
            mutual_matches.append(m)

    return sorted(mutual_matches, key=lambda x: x.distance)



def load_features(npz_file):
    data = np.load(npz_file)
    kps = [cv2.KeyPoint(float(x), float(y), 1) for x, y in data["keypoints"]]
    desc = data["descriptors"]
    return kps, desc, data["keypoints"]


def draw_matches_thick(img1, kps1, img2, kps2, matches, max_display=100, thickness=2):
    """拼接两张图并画匹配，支持加粗线条"""
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


def find_feature_file(features_root, scene, filename, method):
    scene_path = normalize_scene(scene)
    target = filename.replace(".jpg", f".{method}.npz")
    search_dir = features_root / scene_path
    #print(f"🔍 搜索特征: {search_dir}/{target}")
    candidates = list(search_dir.rglob(target))
    if len(candidates) == 0:
        return None
    return candidates[0]

def find_image_file(images_root, scene, filename):
    scene_path = normalize_scene(scene)
    search_dir = images_root / scene_path
    #print(f"🔍 搜索图像: {search_dir}/{filename}")
    candidates = list(search_dir.rglob(filename))
    if len(candidates) == 0:
        return None
    return candidates[0]




def normalize_scene(scene: str) -> str:
    """
    把 pairs_eval.txt 中的 scene 转换成实际目录结构
    - 第一个 '-' 之前保持不变
    - 第二个 '-' 转换成 '/'
    - 之后的 '-' 保持不变
    """
    parts = scene.split("-", 2)  # 最多切两次
    if len(parts) == 1:
        return scene
    elif len(parts) == 2:
        return parts[0] + "/" + parts[1]
    else:
        # 前两段变成路径，后面保持原样
        return parts[0] + "/" + parts[1] + "-" + parts[2]




# ============ 主流程 ============
def main():
    pairs = [line.strip().split(",") for line in open(pairs_file, "r").readlines()[1:]]

    print(f"找到 {len(pairs)} 对图像对")

    for method in methods:
        out_dir = output_root / f"{method}+mnn"
        vis_dir = out_dir / "vis"
        npz_dir = out_dir / "npz"
        vis_dir.mkdir(parents=True, exist_ok=True)
        npz_dir.mkdir(parents=True, exist_ok=True)

        for scene, f1, f2 in pairs:
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
                
            # 加载特征
            kps1, desc1, pts1 = load_features(feat1_file)
            kps2, desc2, pts2 = load_features(feat2_file)

            # 匹配
            matches = match_mnn(desc1, desc2)
            print(f"[{method}] {scene}, {f1}-{f2}: {len(matches)} matches")

            # 保存 npz
            matched_pts1 = np.array([pts1[m.queryIdx] for m in matches], dtype=np.float32)
            matched_pts2 = np.array([pts2[m.trainIdx] for m in matches], dtype=np.float32)
            #out_npz = npz_dir / f"{scene}_{Path(f1).stem}_{Path(f2).stem}.npz"
            # 输出目录层级：out_dir / scene / {npz, vis}
            scene_path = Path(scene.replace("-", "/"))
            scene_npz_dir = npz_dir / scene_path
            scene_vis_dir = vis_dir / scene_path
            scene_npz_dir.mkdir(parents=True, exist_ok=True)
            scene_vis_dir.mkdir(parents=True, exist_ok=True)
            out_npz = scene_npz_dir / f"{Path(f1).stem}_{Path(f2).stem}.npz"
            

            np.savez(out_npz, pts1=matched_pts1, pts2=matched_pts2)

            # 可视化
            img1 = cv2.imread(str(img1_file))
            img2 = cv2.imread(str(img2_file))
            vis_img = draw_matches_thick(img1, kps1, img2, kps2, matches, max_display=10, thickness=2)
            #out_vis = vis_dir / f"{scene}_{Path(f1).stem}_{Path(f2).stem}.jpg"
            out_vis = scene_vis_dir / f"{Path(f1).stem}_{Path(f2).stem}.jpg"
            cv2.imwrite(str(out_vis), vis_img)
            

if __name__ == "__main__":
    main()