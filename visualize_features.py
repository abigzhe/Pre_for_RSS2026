import cv2
import numpy as np
from pathlib import Path

# 输入输出目录
features_root = Path("/data1/home/limingzhe/abigzhe_dinopro/result/randomall/superpoint&sift/features")
image_root = Path("/data1/home/limingzhe/abigzhe_dinopro/result/randomall/images")
output_root = Path("/data1/home/limingzhe/abigzhe_dinopro/result/randomall/superpoint&sift/visualize")
output_root.mkdir(parents=True, exist_ok=True)



def load_keypoints(npz_file: Path):
    data = np.load(npz_file)
    kps = data["keypoints"]
    # 点半径加大，实心显示
    return [cv2.KeyPoint(float(x), float(y), 10) for x, y in kps]

def npz_rel_to_jpg_rel(npz_rel: Path) -> Path:
    """
    把  xxx/yyy/031.superpoint.npz  或  031.sift.npz
    映射为  xxx/yyy/images/031.jpg
    """
    # 去掉 .npz -> "031.superpoint"
    no_npz = npz_rel.with_suffix("")         
    # 去掉 .superpoint / .sift，换成 .jpg
    jpg_name = no_npz.stem.split(".")[0] + ".jpg"  
    # 在倒数第二层路径后插入 "images"
    return no_npz.parent / "images" / jpg_name

def sp_to_sift_path(sp_path: Path) -> Path:
    no_npz = sp_path.with_suffix("")
    return no_npz.with_suffix(".sift.npz")



def draw_label(img, text):
    """在图像顶部加超大字号文字标签"""
    vis = img.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 4.0        # 超大字号
    thickness = 6      # 粗体
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    pad = 30
    # 半透明黑色背景条
    overlay = vis.copy()
    cv2.rectangle(overlay, (0, 0), (tw + 2 * pad, th + 2 * pad),
                  (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, vis, 0.4, 0, vis)
    # 白色文字
    cv2.putText(vis, text, (pad, th + pad - 10), font, scale,
                (255, 255, 255), thickness, cv2.LINE_AA)
    return vis


def draw_keypoints_solid(img, kps, color, radius=6):
    """用实心圆绘制关键点"""
    vis = img.copy()
    for kp in kps:
        x, y = int(kp.pt[0]), int(kp.pt[1])
        cv2.circle(vis, (x, y), radius, color, -1)  # -1 = 实心
    return vis


def visualize_pair(img_path: Path, sp_file: Path, sift_file: Path, out_file: Path):
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"⚠️ 无法读取图像: {img_path}")
        return

    panels = []

    # SuperPoint
    if sp_file.exists():
        sp_kps = load_keypoints(sp_file)
        vis_sp = draw_keypoints_solid(img, sp_kps, color=(255, 0, 255), radius=6)  # 洋红
        vis_sp = draw_label(vis_sp, "SuperPoint")
        panels.append(vis_sp)
    else:
        print(f"⚠️ 缺少 SuperPoint 文件: {sp_file}")

    # SIFT
    if sift_file.exists():
        sift_kps = load_keypoints(sift_file)
        vis_sift = draw_keypoints_solid(img, sift_kps, color=(255, 165, 0), radius=6)  # 橙色
        vis_sift = draw_label(vis_sift, "SIFT")
        panels.append(vis_sift)
    else:
        print(f"⚠️ 缺少 SIFT 文件: {sift_file}")

    if not panels:
        return

    vis = panels[0] if len(panels) == 1 else cv2.hconcat(panels)

    out_file.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_file), vis)
    print(f"✅ 保存可视化对比: {out_file}")


def main():
    sp_files = sorted(features_root.rglob("*.superpoint.npz"))
    print(f"找到 {len(sp_files)} 个 SuperPoint 特征文件")

    for sp_file in sp_files:
        rel_path = sp_file.relative_to(features_root)
        # 正确映射到原图路径：*.superpoint.npz -> *.jpg
        img_path = image_root / npz_rel_to_jpg_rel(rel_path)
        if not img_path.exists():
            print(f"⚠️ 找不到原图: {img_path}")
            continue

        # 找对应的 SIFT 特征文件：*.superpoint.npz -> *.sift.npz
        sift_file = sp_to_sift_path(sp_file)

        out_file = output_root / rel_path.with_suffix("").with_suffix(".compare.png")

        # ✅ 如果已经存在，跳过
        if out_file.exists():
            print(f"⏭️ 已存在，跳过: {out_file}")
            continue

        visualize_pair(img_path, sp_file, sift_file, out_file)


if __name__ == "__main__":
    main()
