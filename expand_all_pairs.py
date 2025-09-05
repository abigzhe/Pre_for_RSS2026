import json
import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation as R

def quaternion_to_rotation_matrix(q):
    """四元数 -> 旋转矩阵 (3x3)"""
    return R.from_quat([q[0], q[1], q[2], q[3]]).as_matrix()

def build_K(focal_length, image_size):
    """构建相机内参矩阵 K"""
    h, w = image_size
    fx = fy = focal_length
    cx = w / 2
    cy = h / 2
    K = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0,  0, 1]
    ], dtype=float)
    return K

def build_T(q, t):
    """构建外参矩阵 T (4x4)"""
    Rmat = quaternion_to_rotation_matrix(q)
    Tmat = np.eye(4)
    Tmat[:3, :3] = Rmat
    Tmat[:3, 3] = np.array(t, dtype=float)
    return Tmat

def expand_pairs(pairs_path, anno_path, output_path):
    """把 pairs.txt 扩展为 38 字段 pairs1.txt"""
    with open(anno_path, "r") as f:
        annotations = json.load(f)

    # 建立 filename -> 相机参数 的映射
    ann_map = {}
    for ann in annotations:
        K = build_K(ann["camera"]["focal_length"], ann["image_size"])
        T = build_T(ann["camera"]["q"], ann["camera"]["t"])
        ann_map[ann["filename"]] = {"K": K, "T": T}

    with open(pairs_path, "r") as f:
        pairs = [line.strip().split() for line in f if line.strip()]

    out_lines = []
    for name0, name1, *rest in pairs:
        rot0, rot1 = 0, 0  # 默认旋转角

        if Path(name0).name not in ann_map or Path(name1).name not in ann_map:
            print(f"⚠️ 跳过 {name0}, {name1}，annotations.json 里没找到")
            continue

        K0, T0 = ann_map[Path(name0).name]["K"], ann_map[Path(name0).name]["T"]
        K1, T1 = ann_map[Path(name1).name]["K"], ann_map[Path(name1).name]["T"]

        # 相对位姿
        T_0to1 = T1 @ np.linalg.inv(T0)

        # 展平为字符串
        K0_str = " ".join(map(str, K0.reshape(-1).tolist()))
        K1_str = " ".join(map(str, K1.reshape(-1).tolist()))
        T_str  = " ".join(map(str, T_0to1.reshape(-1).tolist()))

        line = f"{name0} {name1} {rot0} {rot1} {K0_str} {K1_str} {T_str}"
        out_lines.append(line)

    if out_lines:
        with open(output_path, "w") as f:
            f.write("\n".join(out_lines))
        print(f"✅ 已生成 {output_path} ({len(out_lines)} 行)")
    else:
        print(f"⚠️ {pairs_path} 没有生成任何行")

def expand_all(root_dir="navi_v1.0"):
    """递归处理整个 NAVI 目录"""
    root = Path(root_dir)
    pairs_files = list(root.rglob("pairs.txt"))
    print(f"🔍 找到 {len(pairs_files)} 个 pairs.txt")

    for pf in pairs_files:
        anno_path = pf.parent / "annotations.json"
        if not anno_path.exists():
            print(f"❌ 跳过 {pf}，缺少 annotations.json")
            continue
        output_path = pf.parent / "pairs1.txt"
        try:
            expand_pairs(pf, anno_path, output_path)
        except Exception as e:
            print(f"❌ 处理 {pf} 出错: {e}")

if __name__ == "__main__":
    expand_all("navi_v1.0")
