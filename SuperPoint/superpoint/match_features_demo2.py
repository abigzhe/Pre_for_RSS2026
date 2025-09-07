import cv2
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from pathlib import Path
from superpoint.settings import EXPER_PATH


# 固定参数
weights_name = "/data1/home/limingzhe/abigzhe_dinopro/SuperPoint/pretrained_models/sp_v6"
input_dir = Path("/data1/home/limingzhe/abigzhe_dinopro/result/randomall/images")
max_size = 1200
output_dir = Path("/data1/home/limingzhe/abigzhe_dinopro/result/randomall/superpoint&sift/features")
output_dir.mkdir(parents=True, exist_ok=True)


def extract_superpoint_keypoints_and_descriptors(keypoint_map, descriptor_map,
                                                 keep_k_points=1000):
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


def resize_keep_aspect(img, max_size=1200):
    h, w = img.shape[:2]
    scale = min(max_size / max(h, w), 1.0)  # 只缩小不放大
    new_w, new_h = int(w * scale), int(h * scale)
    return cv2.resize(img, (new_w, new_h))


def preprocess_image(img_file, max_size):
    img = cv2.imread(img_file, cv2.IMREAD_COLOR)
    assert img is not None, f"无法读取图像: {img_file}"
    h, w = img.shape[:2]
    scale = min(max_size / max(h, w), 1.0)  # 缩放比例
    new_w, new_h = int(w * scale), int(h * scale)
    img_resized = cv2.resize(img, (new_w, new_h))
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    gray = gray.astype(np.float32) / 255.0
    gray = np.expand_dims(gray, axis=-1)  # [H,W,1]
    return gray, img, scale   # 返回原图(img)、缩放比例


def save_features(out_file, keypoints, descriptors):
    out_file.parent.mkdir(parents=True, exist_ok=True)  # 确保目录存在
    kps = np.array([[kp.pt[0], kp.pt[1]] for kp in keypoints], dtype=np.float32)
    np.savez(out_file, keypoints=kps, descriptors=descriptors)
    print(f"✅ 保存特征到 {out_file}")


def extract_SIFT_keypoints_and_descriptors(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    if hasattr(cv2, "SIFT_create"):
        sift = cv2.SIFT_create()
    else:
        sift = cv2.xfeatures2d.SIFT_create()
    kp, desc = sift.detectAndCompute(gray, None)
    return kp, desc


def main():
    # 模型路径
    candidate = Path(weights_name)
    if candidate.is_dir() and (candidate / "saved_model.pb").exists():
        weights_dir = candidate
    else:
        weights_dir = Path(EXPER_PATH, 'saved_models', weights_name)
    weights_dir = weights_dir.resolve()
    assert (weights_dir / "saved_model.pb").exists(), f"找不到模型: {weights_dir}"

    # 遍历所有图片
    img_paths = sorted(input_dir.rglob("*.jpg"))
    print(f"找到 {len(img_paths)} 张图片")

    graph = tf.Graph()
    with tf.Session(graph=graph) as sess:
        tf.saved_model.loader.load(
            sess,
            [tf.saved_model.tag_constants.SERVING],
            str(weights_dir)
        )

        input_img_tensor = graph.get_tensor_by_name('superpoint/image:0')
        output_prob_nms_tensor = graph.get_tensor_by_name('superpoint/prob_nms:0')
        output_desc_tensors = graph.get_tensor_by_name('superpoint/descriptors:0')

        for img_file in img_paths:
            gray, img_orig, scale = preprocess_image(img_file, max_size)

            # ---- SuperPoint ----
            prob, descmap = sess.run(
                [output_prob_nms_tensor, output_desc_tensors],
                feed_dict={input_img_tensor: np.expand_dims(gray, 0)}
            )
            keypoint_map = np.squeeze(prob)
            descriptor_map = np.squeeze(descmap)

            sp_kps, sp_desc = extract_superpoint_keypoints_and_descriptors(
                keypoint_map, descriptor_map, keep_k_points=1000
            )

            # ⚡ 坐标映射回原始图像
            for kp in sp_kps:
                kp.pt = (kp.pt[0] / scale, kp.pt[1] / scale)

            # ✅ 去掉 "images" 文件夹
            rel_path = img_file.relative_to(input_dir)
            if "images" in rel_path.parts:
                parts = list(rel_path.parts)
                parts.remove("images")
                rel_path = Path(*parts).with_suffix("")
            else:
                rel_path = rel_path.with_suffix("")

            out_file = output_dir / rel_path

            save_features(out_file.with_suffix(".superpoint.npz"), sp_kps, sp_desc)

            # ---- SIFT ----
            sift_kps, sift_desc = extract_SIFT_keypoints_and_descriptors(img_orig)
            if sift_desc is None:
                sift_desc = np.empty((0, 128), dtype=np.float32)
                sift_kps = []
            save_features(out_file.with_suffix(".sift.npz"), sift_kps, sift_desc)


if __name__ == "__main__":
    main()
