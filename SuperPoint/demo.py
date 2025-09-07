import cv2
import numpy as np
import torch
from superpoint.models.superpoint import SuperPoint  # 具体路径可能略有不同

# --- 加载 SuperPoint ---
def load_superpoint(weights_path="superpoint_v1.pth", use_gpu=True):
    config = {
        'descriptor_dim': 256,
        'nms_radius': 4,
        'keypoint_threshold': 0.005,
        'max_keypoints': 1024
    }
    device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
    sp = SuperPoint(config).to(device)
    sp.load_state_dict(torch.load(weights_path, map_location=device))
    sp.eval()
    return sp, device

# --- 提取关键点和描述子 ---
def extract_features(img_path, model, device):
    # 读图，转灰度
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = img.astype(np.float32) / 255.
    inp = torch.from_numpy(img)[None, None].to(device)

    with torch.no_grad():
        pred = model({'image': inp})

    # 关键点
    keypoints = pred['keypoints'][0].cpu().numpy()
    # 描述子
    descriptors = pred['descriptors'][0].cpu().numpy().T

    return keypoints, descriptors

if __name__ == "__main__":
    model, device = load_superpoint("superpoint_v1.pth")

    kpts, desc = extract_features("example.jpg", model, device)

    print("检测到关键点数:", len(kpts))
    print("关键点示例:", kpts[:5])
    print("描述子维度:", desc.shape)  # (N, 256)

    # 可视化
    img = cv2.imread("example.jpg")
    for x, y in kpts.astype(int):
        cv2.circle(img, (x, y), 2, (0, 255, 0), -1)
    cv2.imshow("SuperPoint Keypoints", img)
    cv2.waitKey(0)
