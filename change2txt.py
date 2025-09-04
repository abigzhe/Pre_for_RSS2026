import os
import json

root_dir = "navi_v1.0"
images_file = "images.txt"
cameras_file = "cameras.txt"

image_id = 1
camera_id = 1
camera_models = {}  # key: (w,h,f) -> camera_id

with open(images_file, "w") as f_images, open(cameras_file, "w") as f_cameras:
    for subdir, _, files in os.walk(root_dir):
        if "annotations.json" not in files:
            continue

        ann_path = os.path.join(subdir, "annotations.json")
        with open(ann_path, "r") as f:
            annotations = json.load(f)

        for entry in annotations:
            q = entry["camera"]["q"]  # [qw, qx, qy, qz]
            t = entry["camera"]["t"]  # 相机中心 (world 坐标系)
            w, h = entry["image_size"]
            f = entry["camera"]["focal_length"]
            fname = os.path.join(subdir, entry["filename"])

            # 确保相机模型唯一
            cam_key = (w, h, f)
            if cam_key not in camera_models:
                cx, cy = w / 2.0, h / 2.0
                f_cameras.write(f"{camera_id} PINHOLE {w} {h} {f} {f} {cx} {cy}\n")
                camera_models[cam_key] = camera_id
                cam_id = camera_id
                camera_id += 1
            else:
                cam_id = camera_models[cam_key]

            # 写入 images.txt
            f_images.write(
                f"{image_id} {q[0]} {q[1]} {q[2]} {q[3]} "
                f"{t[0]} {t[1]} {t[2]} {cam_id} {fname}\n"
            )
            image_id += 1
