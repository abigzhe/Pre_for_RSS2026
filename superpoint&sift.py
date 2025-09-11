#!/usr/bin/env python3
import os

# ====== 强制禁用 GPU，在 CPU 上跑 ======
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# 禁用 XLA（避免 JIT 编译和 libdevice 依赖）
os.environ["TF_XLA_FLAGS"] = "--tf_xla_auto_jit=0 --tf_xla_enable_xla_devices=false"

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import cv2
import numpy as np
from pathlib import Path
import time
from tqdm import tqdm

# ================ 配置 ================
weights_dir = Path("/data1/home/limingzhe/abigzhe_dinopro/SuperPoint/pretrained_models/sp_v6")
input_dir   = Path("/data1/home/limingzhe/abigzhe_dinopro/navi_v1.0")
output_dir  = Path("/data1/home/limingzhe/abigzhe_dinopro/result/all/superpoint/pointmask_cpu")
vis_dir     = output_dir / "vis"
output_dir.mkdir(parents=True, exist_ok=True)
vis_dir.mkdir(parents=True, exist_ok=True)

POINT_RADIUS = 6
FONT_SCALE = 3.0
FONT_THICK = 4
SIFT_MAX_FEATURES = 10000
MAX_SP_INPUT_LONGER_EDGE = 768

# ================ 工具函数 ================
def extract_superpoint_keypoints_and_descriptors(keypoint_map, descriptor_map, keep_k_points=1000):
    def select_k_best(points, k):
        idx = points[:, 2].argsort()
        pts_sorted = points[idx, :2]
        start = min(k, points.shape[0])
        return pts_sorted[-start:, :]

    ys, xs = np.where(keypoint_map > 0)
    if len(xs) == 0:
        return [], np.empty((0, descriptor_map.shape[-1]), dtype=np.float32)

    prob = keypoint_map[ys, xs]
    pts = np.stack([ys, xs, prob], axis=-1)
    pts = select_k_best(pts, keep_k_points).astype(int)
    desc = descriptor_map[pts[:, 0], pts[:, 1]]
    kps = [cv2.KeyPoint(int(x), int(y), 1) for y, x in pts]
    return kps, desc.astype(np.float32)

def preprocess_image(img_file, max_size=MAX_SP_INPUT_LONGER_EDGE):
    img_orig = cv2.imread(img_file, cv2.IMREAD_COLOR)
    if img_orig is None:
        raise RuntimeError(f"无法读取图像: {img_file}")
    h0, w0 = img_orig.shape[:2]
    longer = max(h0, w0)
    scale = 1.0
    img_for_model = img_orig
    if longer > max_size:
        scale = float(max_size) / float(longer)
        new_w = int(round(w0 * scale))
        new_h = int(round(h0 * scale))
        img_for_model = cv2.resize(img_orig, (new_w, new_h), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(img_for_model, cv2.COLOR_BGR2GRAY)
    gray = gray.astype(np.float32) / 255.0
    gray = np.expand_dims(gray, axis=-1)
    return gray, img_for_model, img_orig, scale

def save_features(out_file, keypoints, descriptors):
    out_file.parent.mkdir(parents=True, exist_ok=True)
    if len(keypoints) == 0:
        kps = np.empty((0,2), dtype=np.float32)
    else:
        kps = np.array([[kp.pt[0], kp.pt[1]] for kp in keypoints], dtype=np.float32)
    np.savez(out_file, keypoints=kps, descriptors=descriptors)
    print(f"  -> 保存: {out_file}")

def extract_SIFT_keypoints_and_descriptors(img_bgr, max_features=SIFT_MAX_FEATURES):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    if hasattr(cv2, "SIFT_create"):
        sift = cv2.SIFT_create(nfeatures=max_features)
    else:
        sift = cv2.xfeatures2d.SIFT_create(nfeatures=max_features)
    kp, desc = sift.detectAndCompute(gray, None)
    if kp is None:
        return [], np.empty((0, 128), dtype=np.float32)
    if desc is None:
        desc = np.empty((0, 128), dtype=np.float32)
    return kp, desc

def visualize_and_save(img_bgr, sp_kps, sift_kps, out_path):
    img_sp = img_bgr.copy()
    img_sift = img_bgr.copy()
    for kp in sp_kps:
        x, y = int(round(kp.pt[0])), int(round(kp.pt[1]))
        cv2.circle(img_sp, (x, y), radius=POINT_RADIUS, color=(0, 255, 0), thickness=-1)
    for kp in sift_kps:
        x, y = int(round(kp.pt[0])), int(round(kp.pt[1]))
        cv2.circle(img_sift, (x, y), radius=POINT_RADIUS, color=(0, 0, 255), thickness=-1)
    h = max(img_sp.shape[0], img_sift.shape[0])
    w = img_sp.shape[1] + img_sift.shape[1]
    vis = np.zeros((h, w, 3), dtype=np.uint8)
    vis[:img_sp.shape[0], :img_sp.shape[1]] = img_sp
    vis[:img_sift.shape[0], img_sp.shape[1]:] = img_sift
    cv2.putText(vis, "SuperPoint", (30, 100),
                cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, (0,0,0), FONT_THICK+2, cv2.LINE_AA)
    cv2.putText(vis, "SuperPoint", (30, 100),
                cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, (255,255,255), FONT_THICK, cv2.LINE_AA)
    cv2.putText(vis, "SIFT", (img_sp.shape[1] + 30, 100),
                cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, (0,0,0), FONT_THICK+2, cv2.LINE_AA)
    cv2.putText(vis, "SIFT", (img_sp.shape[1] + 30, 100),
                cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, (255,255,255), FONT_THICK, cv2.LINE_AA)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), vis)
    print(f"  -> 可视化保存: {out_path}")

def filter_by_mask(kps, desc, mask):
    """按掩膜过滤特征点和描述子"""
    if mask is None or len(kps) == 0:
        return kps, desc
    filtered_kps = []
    filtered_desc = []
    for kp, d in zip(kps, desc):
        x, y = int(round(kp.pt[0])), int(round(kp.pt[1]))
        if 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1] and mask[y, x] > 0:
            filtered_kps.append(kp)
            filtered_desc.append(d)
    if len(filtered_desc) == 0:
        filtered_desc = np.empty((0, desc.shape[1]), dtype=np.float32)
    else:
        filtered_desc = np.vstack(filtered_desc)
    return filtered_kps, filtered_desc

# ================ 主流程 ================
def main():
    print("TF version:", tf.__version__)
    print("GPUs (should be empty on CPU):", tf.config.list_physical_devices('GPU'))

    img_paths = sorted(input_dir.rglob("*.jpg"))
    img_paths = [p for p in img_paths if "3d_scan" not in p.parts]
    print(f"找到 {len(img_paths)} 张图片（已忽略 3d_scan）")

    vis_saved_count = 0

    graph = tf.Graph()
    config = tf.ConfigProto(device_count={'GPU': 0})  # 强制 CPU
    config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.OFF

    with tf.Session(graph=graph, config=config) as sess:
        tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], str(weights_dir))
        input_img_tensor = graph.get_tensor_by_name('superpoint/image:0')
        output_prob_nms_tensor = graph.get_tensor_by_name('superpoint/prob_nms:0')
        output_desc_tensors = graph.get_tensor_by_name('superpoint/descriptors:0')

        for idx, img_file in enumerate(tqdm(img_paths, desc="Processing images")):
            t0 = time.time()
            try:
                gray_for_model, img_resized, img_orig, scale = preprocess_image(str(img_file), max_size=MAX_SP_INPUT_LONGER_EDGE)
            except Exception as e:
                print(f"跳过（无法读图）: {img_file}  错误: {e}")
                continue

            rel_path = img_file.relative_to(input_dir).with_suffix("")
            out_file = output_dir / rel_path
            sp_path = out_file.with_suffix(".superpoint.npz")
            sift_path = out_file.with_suffix(".sift.npz")

            if sp_path.exists() and sift_path.exists():
                print(f"✅ [{idx+1}/{len(img_paths)}] 已存在结果，跳过 {img_file}")
                continue

            # ---- SuperPoint 推理 ----
            prob, descmap = sess.run(
                [output_prob_nms_tensor, output_desc_tensors],
                feed_dict={input_img_tensor: np.expand_dims(gray_for_model, 0)}
            )
            keypoint_map = np.squeeze(prob)
            descriptor_map = np.squeeze(descmap)
            sp_kps, sp_desc = extract_superpoint_keypoints_and_descriptors(keypoint_map, descriptor_map)

            if scale != 1.0 and len(sp_kps) > 0:
                for kp in sp_kps:
                    kp.pt = (kp.pt[0] / scale, kp.pt[1] / scale)

            # ---- SIFT ----
            sift_kps, sift_desc = extract_SIFT_keypoints_and_descriptors(img_orig, max_features=SIFT_MAX_FEATURES)

            # ---- 读取掩膜并过滤 ----
            mask_file = img_file.parent.parent / "masks" / (img_file.stem + ".png")
            mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE) if mask_file.exists() else None
            sp_kps, sp_desc = filter_by_mask(sp_kps, sp_desc, mask)
            sift_kps, sift_desc = filter_by_mask(sift_kps, sift_desc, mask)

            # ---- 保存特征 ----
            save_features(sp_path, sp_kps, sp_desc)
            save_features(sift_path, sift_kps, sift_desc)

            # ---- 只保存前 5 张可视化 ----
            if vis_saved_count < 5:
                vis_img_path = vis_dir / (rel_path.name + "_compare.jpg")
                visualize_and_save(img_orig, sp_kps, sift_kps, vis_img_path)
                vis_saved_count += 1

            print(f"✅ [{idx+1}/{len(img_paths)}] 完成 {img_file} | ⏱ {time.time()-t0:.2f}s")

    print("全部处理完成。")

if __name__ == "__main__":
    main()
