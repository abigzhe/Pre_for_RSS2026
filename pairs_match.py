import os
import json
import itertools

root_dir = "navi_v1.0"

for subdir, _, files in os.walk(root_dir):
    if "annotations.json" not in files:
        continue

    ann_path = os.path.join(subdir, "annotations.json")
    with open(ann_path, "r") as f:
        annotations = json.load(f)

    # --- 输出文件路径 ---
    images_file = os.path.join(subdir, "images.txt")
    cameras_file = os.path.join(subdir, "cameras.txt")
    pairs_file = os.path.join(subdir, "pairs.txt")

    # --- cameras.txt ---
    w, h = annotations[0]["image_size"]
    f = annotations[0]["camera"]["focal_length"]
    cx, cy = w / 2.0, h / 2.0
    with open(cameras_file, "w") as f_cam:
        f_cam.write(f"1 PINHOLE {w} {h} {f} {f} {cx} {cy}\n")

    # 图像目录（即 subdir/images/）
    images_dir = os.path.join(subdir, "images")

    # --- images.txt ---
    with open(images_file, "w") as f_img:
        image_id = 1
        for entry in annotations:
            q = entry["camera"]["q"]  # [qw, qx, qy, qz]
            t = entry["camera"]["t"]  # [tx, ty, tz]
            fname = os.path.join(images_dir, entry["filename"])
            f_img.write(
                f"{image_id} {q[0]} {q[1]} {q[2]} {q[3]} "
                f"{t[0]} {t[1]} {t[2]} 1 {fname}\n"
            )
            image_id += 1

    # --- pairs.txt ---
    image_paths = [os.path.join(images_dir, ann["filename"]) for ann in annotations]
    pairs = list(itertools.combinations(image_paths, 2))
    with open(pairs_file, "w") as f_pairs:
        for img1, img2 in pairs:
            f_pairs.write(f"{img1} {img2}\n")

    print(f"✅ Generated in {subdir}: images.txt, cameras.txt, pairs.txt")
