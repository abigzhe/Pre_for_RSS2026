from pathlib import Path
import argparse
import json
import random
import numpy as np
import matplotlib.cm as cm
import torch

# 导入 SuperGlue 相关模块
from models.matching import Matching
from models.utils import (
    compute_pose_error,         # 姿态误差计算
    compute_epipolar_error,     # 极线误差计算
    estimate_pose,              # 用匹配点估计相机姿态
    make_matching_plot,         # 匹配可视化
    error_colormap,             # 错误可视化配色
    AverageTimer,               # 计时工具
    pose_auc,                   # AUC 评估指标
    read_image,                 # 图像读取与预处理
    rotate_intrinsics,          # 相机内参旋转
    rotate_pose_inplane,        # 相机外参旋转
    scale_intrinsics            # 相机内参缩放
)

# 关闭梯度计算，加速推理（只做推理，不训练）
torch.set_grad_enabled(False)

if __name__ == '__main__':
    # ========== 命令行参数解析 ==========
    parser = argparse.ArgumentParser(
        description='Batch run SuperGlue matching across all pairs.txt under navi_v1.0',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--root_dir', type=str, default='navi_v1.0',
        help='包含 pairs.txt 的根目录')
    parser.add_argument('--max_length', type=int, default=-1,
        help='限制每个 pairs.txt 中处理的最大图像对数')
    # 默认缩放策略：保持长边 = 1200（与 NAVI 论文一致）
    parser.add_argument('--resize', type=int, nargs='+', default=[1200],
        help='推理前是否对图像缩放。如果给一个数字，则缩放最长边；如果给两个数字，则缩放到固定宽高；-1 表示不缩放')
    parser.add_argument('--resize_float', action='store_true',
        help='是否在归一化前进行 float 缩放')
    # SuperGlue 参数
    parser.add_argument('--superglue', choices={'indoor', 'outdoor'}, default='indoor')
    parser.add_argument('--max_keypoints', type=int, default=1024)
    parser.add_argument('--keypoint_threshold', type=float, default=0.005)
    parser.add_argument('--nms_radius', type=int, default=4)
    parser.add_argument('--sinkhorn_iterations', type=int, default=20)
    parser.add_argument('--match_threshold', type=float, default=0.2)
    # 输出与可视化参数
    parser.add_argument('--viz', action='store_true', help='是否保存可视化结果')
    parser.add_argument('--eval', action='store_true', help='是否进行姿态评估')
    parser.add_argument('--fast_viz', action='store_true', help='是否使用 OpenCV 快速绘图')
    parser.add_argument('--cache', action='store_true', help='是否启用缓存（已有结果则跳过）')
    parser.add_argument('--show_keypoints', action='store_true', help='是否在可视化中显示关键点')
    parser.add_argument('--viz_extension', type=str, default='png', choices=['png', 'pdf'], help='可视化文件格式')
    parser.add_argument('--opencv_display', action='store_true', help='是否实时显示匹配结果')
    parser.add_argument('--shuffle', action='store_true', help='是否打乱输入对顺序')
    parser.add_argument('--force_cpu', action='store_true', help='是否强制使用 CPU')

    # === 稠密召回（Dense-Recall@τ）相关参数 ===
    parser.add_argument('--dense_eval', action='store_true', help='是否计算 Dense-Recall@τ（需要稠密真值点对）')
    parser.add_argument('--dense_thresh', type=float, default=15.0, help='Dense-Recall 的像素阈值 τ')
    parser.add_argument('--dense_mode', type=str, default='npz', choices=['npz', 'module'],
                        help='稠密真值的来源：npz=从预计算的 npz 读取；module=从外部模块函数动态生成')
    parser.add_argument('--dense_gt_dir', type=str, default=None,
                        help='当 dense_mode=npz 时：存放每对图像稠密真值 npz 的目录（文件需包含 gt0, gt1 数组）')
    parser.add_argument('--dense_module', type=str, default='test',
                        help='当 dense_mode=module 时：包含 project_and_filter_sample_coordinates 的模块名（例如 test.py）')
    parser.add_argument('--dense_func', type=str, default='project_and_filter_sample_coordinates',
                        help='当 dense_mode=module 时：生成投影坐标的函数名（返回 {point_id: (y,x)} 字典）')
    parser.add_argument('--dense_samples', type=int, default=5000,
                        help='当 dense_mode=module 时：用于稠密评估的采样点数量（取决于外部实现）')

    opt = parser.parse_args([])  # 强制使用默认参数（无需命令行输入）
    opt.cache = False  # 默认强制覆盖旧结果
    print(opt)

    # 默认强制开启可视化
    opt.viz = True
    print("⚙️ 已强制启用可视化模式")

    # ========== 加载 SuperPoint + SuperGlue 模型 ==========
    device = 'cuda' if torch.cuda.is_available() and not opt.force_cpu else 'cpu'
    print(f'Running inference on device {device}')
    config = {
        'superpoint': {
            'nms_radius': opt.nms_radius,
            'keypoint_threshold': opt.keypoint_threshold,
            'max_keypoints': opt.max_keypoints
        },
        'superglue': {
            'weights': opt.superglue,
            'sinkhorn_iterations': opt.sinkhorn_iterations,
            'match_threshold': opt.match_threshold,
        }
    }
    matching = Matching(config).eval().to(device)

    # === Dense-Recall 支持函数 ===
    def compute_dense_recall(mkpts0: np.ndarray, mkpts1: np.ndarray,
                              gt0: np.ndarray, gt1: np.ndarray,
                              thresh: float = 15.0) -> float:
        """计算 Dense-Recall@thresh
        定义：对每个预测匹配 (mkpts0[i] -> mkpts1[i])，
        在 gt0 上找最近的真值点 j，并拿它在另一张图的投影 gt1[j] 与 mkpts1[i] 比较，
        统计像素误差 <= thresh 的比例。
        """
        if mkpts0.size == 0 or gt0.size == 0:
            return 0.0
        # 最近邻：mkpts0 → gt0
        dists = np.linalg.norm(mkpts0[:, None, :] - gt0[None, :, :], axis=-1)
        nn_idx = np.argmin(dists, axis=1)
        reproj_err = np.linalg.norm(mkpts1 - gt1[nn_idx], axis=-1)
        return float(np.mean(reproj_err <= thresh))

    def load_annotations(json_path: Path):
        with open(json_path, 'r') as f:
            return json.load(f)

    def build_filename_to_ann(annotations):
        # 建立 filename → annotation 的索引
        mapping = {}
        for ann in annotations:
            # 兼容 '000.jpg' / '000.png' 等
            filename = ann.get('filename')
            mapping[filename] = ann
        return mapping

    def try_dense_from_npz(scene_dir: Path, stem0: str, stem1: str):
        """当 dense_mode=npz：尝试在 --dense_gt_dir 或当前场景下读取预计算 npz
        期望文件包含数组 gt0, gt1（形状 [N,2]，像素坐标 yx 或 xy 需与下方一致）。
        命名规范建议：{stem0}_{stem1}_dense_gt.npz
        """
        candidates = []
        if opt.dense_gt_dir is not None:
            base = Path(opt.dense_gt_dir)
            candidates.append(base / f'{stem0}_{stem1}_dense_gt.npz')
            candidates.append(base / f'{stem1}_{stem0}_dense_gt.npz')
        # 也尝试场景目录下的 dump_dense_gt/
        candidates.append(scene_dir / 'dump_dense_gt' / f'{stem0}_{stem1}_dense_gt.npz')
        candidates.append(scene_dir / 'dump_dense_gt' / f'{stem1}_{stem0}_dense_gt.npz')
        for p in candidates:
            if p.exists():
                data = np.load(p)
                gt0, gt1 = data['gt0'], data['gt1']
                # 统一到 (x,y) 顺序
                if gt0.shape[1] == 2 and np.mean(gt0[:,0]) < np.mean(gt0[:,1]):
                    # 无法可靠判断坐标顺序，这里假设已为 (x,y)。
                    pass
                return gt0, gt1
        return None, None

    def try_dense_from_module(scene_dir: Path, ann_map: dict, name0: str, name1: str,
                              image0: np.ndarray, image1: np.ndarray):
        """当 dense_mode=module：通过外部模块函数生成稠密投影坐标
        需要一个函数签名类似：coords = fn(annotation, sampled_points, image)
        返回 {point_id: (y, x)} 或 (x, y)，我们会统一为 (x,y)。
        注意：采样点与渲染深度/可见性由外部模块负责（复用你的 test.py）。
        """
        try:
            mod = __import__(opt.dense_module)
            fn = getattr(mod, opt.dense_func)
        except Exception as e:
            print(f"[Dense] 无法导入模块或函数：{opt.dense_module}.{opt.dense_func}，跳过。原因：{e}")
            return None, None
        # 加载 annotations.json
        json_path = scene_dir / 'annotations.json'
        if not json_path.exists():
            print(f"[Dense] 缺少 {json_path}，跳过 dense 计算。")
            return None, None
        anns = load_annotations(json_path)
        ann_map2 = build_filename_to_ann(anns)
        ann0 = ann_map2.get(Path(name0).name)
        ann1 = ann_map2.get(Path(name1).name)
        if ann0 is None or ann1 is None:
            print(f"[Dense] annotations.json 中找不到 {name0} 或 {name1}，跳过。")
            return None, None
        # 外部模块负责提供 sampled_points 的来源；如果需要，我们尝试提供数量参数
        # 这里假设外部模块内部会根据 scene 或 mesh 数据生成采样点
        try:
            coords0 = fn(annotation=ann0, image=image0, num_samples=opt.dense_samples)
            coords1 = fn(annotation=ann1, image=image1, num_samples=opt.dense_samples)
        except TypeError:
            # 兼容不同签名：project_and_filter_sample_coordinates(annotation, sampled_points, image)
            try:
                if hasattr(mod, 'sample_points_from_mesh') and hasattr(mod, 'get_mesh_triangles'):
                    mesh_triangles = mod.get_mesh_triangles(scene_dir)
                    sampled_points, _ = mod.sample_points_from_mesh(mesh_triangles, opt.dense_samples)
                    coords0 = fn(mesh_triangles=mesh_triangles, annotation=ann0,
                                 sampled_points=sampled_points, image=image0)
                    coords1 = fn(mesh_triangles=mesh_triangles, annotation=ann1,
                                 sampled_points=sampled_points, image=image1)
                else:
                    print('[Dense] 外部函数签名不匹配，且缺少 mesh 支持，跳过 Dense 计算。')
                    return None, None
            except Exception as e:
                print(f"[Dense] 调用外部函数失败：{e}")
                return None, None
        # 统一 coords 为 (x,y) np.ndarray
        def dict_to_xy(coords: dict):
            if coords is None or len(coords) == 0:
                return np.empty((0,2), dtype=float)
            # coords: id -> (y,x) or (x,y)
            arr = np.array(list(coords.values()), dtype=float)
            # 尝试判定 (y,x) → (x,y)
            # 简单启发：图像形状可用来判断，但此处不强制交换，假设外部为 (x,y)
            return arr
        # 找共同 id
        ids0 = set(coords0.keys()) if isinstance(coords0, dict) else set()
        ids1 = set(coords1.keys()) if isinstance(coords1, dict) else set()
        common = list(ids0 & ids1)
        if len(common) == 0:
            return None, None
        gt0 = np.array([coords0[i] for i in common], dtype=float)
        gt1 = np.array([coords1[i] for i in common], dtype=float)
        # 若为 (y,x)，交换到 (x,y)
        if gt0.shape[0] > 0 and image0 is not None:
            H0, W0 = image0.shape[0], image0.shape[1]
            if np.mean(gt0[:,0]) > H0 * 0.6:  # 像 y 值过大时，可能是 (y,x)
                gt0 = gt0[:, [1,0]]
        if gt1.shape[0] > 0 and image1 is not None:
            H1, W1 = image1.shape[0], image1.shape[1]
            if np.mean(gt1[:,0]) > H1 * 0.6:
                gt1 = gt1[:, [1,0]]
        return gt0, gt1

    # ========== 遍历 navi_v1.0 下所有 pairs.txt ==========
    root_dir = Path(opt.root_dir)
    pairs_files = list(root_dir.rglob("pairs.txt"))
    print(f"在 {root_dir} 下找到 {len(pairs_files)} 个 pairs.txt")

    for pf in pairs_files:
        print(f"========== 正在处理 {pf} ==========")

        # 输出目录 = 当前 pairs.txt 所在目录下的 dump_match_pairs
        output_dir = pf.parent / "dump_match_pairs"
        output_dir.mkdir(exist_ok=True, parents=True)
        print(f"结果将保存到 {output_dir}")

        # 读取 pairs.txt 中的图像对
        pairs = []
        with open(pf, 'r') as f:
            for line in f:
                if line.strip() == "":
                    continue
                pairs.append(line.strip().split())

        # 可选：限制数量
        if opt.max_length > -1:
            pairs = pairs[:np.min([len(pairs), opt.max_length])]
        # 可选：打乱顺序
        if opt.shuffle:
            random.Random(0).shuffle(pairs)

        # 初始化计时器
        timer = AverageTimer(newline=True)

        # ========== 遍历该 pairs.txt 中的所有图像对 ==========
        for i, pair in enumerate(pairs):
            name0, name1 = pair[:2]  # 图像路径
            stem0, stem1 = Path(name0).stem, Path(name1).stem

            # 输出文件路径
            matches_path = output_dir / f"{stem0}_{stem1}_matches.npz"
            eval_path = output_dir / f"{stem0}_{stem1}_evaluation.npz"
            viz_path = output_dir / f"{stem0}_{stem1}_matches.{opt.viz_extension}"
            viz_eval_path = output_dir / f"{stem0}_{stem1}_evaluation.{opt.viz_extension}"

            # 判断是否需要重新计算 / 可视化
            do_match, do_eval, do_viz, do_viz_eval = True, opt.eval, opt.viz, opt.eval and opt.viz
            if opt.cache:
                if matches_path.exists():
                    do_match = False
                if opt.eval and eval_path.exists():
                    do_eval = False
                if opt.viz and viz_path.exists():
                    do_viz = False
                if opt.viz and opt.eval and viz_eval_path.exists():
                    do_viz_eval = False

            # 如果所有结果已存在，则直接跳过
            if not (do_match or do_eval or do_viz or do_viz_eval):
                timer.print(f'Finished pair {i+1:5} of {len(pairs):5}')
                continue

            # 旋转角度（若 pairs.txt 中有提供）
            rot0, rot1 = (int(pair[2]), int(pair[3])) if len(pair) >= 5 else (0, 0)

            # 读取图像，得到：原图、张量输入、缩放比例
            image0, inp0, scales0 = read_image(Path(name0), device, opt.resize, rot0, opt.resize_float)
            image1, inp1, scales1 = read_image(Path(name1), device, opt.resize, rot1, opt.resize_float)
            if image0 is None or image1 is None:
                print(f"读取失败: {name0} {name1}")
                continue

            # ========== 执行 SuperGlue 匹配 ==========
            if do_match:
                pred = matching({'image0': inp0, 'image1': inp1})
                pred = {k: v[0].cpu().numpy() for k, v in pred.items()}
                kpts0, kpts1 = pred['keypoints0'], pred['keypoints1']
                matches, conf = pred['matches0'], pred['matching_scores0']
                out_matches = {
                    'keypoints0': kpts0, 'keypoints1': kpts1,
                    'matches': matches, 'match_confidence': conf
                }
                np.savez(str(matches_path), **out_matches)
            else:
                # 如果已有缓存，直接加载
                results = np.load(matches_path)
                kpts0, kpts1 = results['keypoints0'], results['keypoints1']
                matches, conf = results['matches'], results['match_confidence']

            # 提取有效匹配点（matches > -1 表示匹配成功）
            valid = matches > -1
            mkpts0, mkpts1, mconf = kpts0[valid], kpts1[matches[valid]], conf[valid]

            # ========== 姿态评估 & 稠密召回计算 ==========
            if do_eval:
                assert len(pair) == 38, 'Eval 模式需要 pairs.txt 每行有 38 个字段'
                K0 = np.array(pair[4:13]).astype(float).reshape(3, 3)
                K1 = np.array(pair[13:22]).astype(float).reshape(3, 3)
                T_0to1 = np.array(pair[22:]).astype(float).reshape(4, 4)

                # 缩放内参
                K0 = scale_intrinsics(K0, scales0)
                K1 = scale_intrinsics(K1, scales1)

                # 考虑旋转
                if rot0 != 0 or rot1 != 0:
                    cam0_T_w = np.eye(4)
                    cam1_T_w = T_0to1
                    if rot0 != 0:
                        K0 = rotate_intrinsics(K0, image0.shape, rot0)
                        cam0_T_w = rotate_pose_inplane(cam0_T_w, rot0)
                    if rot1 != 0:
                        K1 = rotate_intrinsics(K1, image1.shape, rot1)
                        cam1_T_w = rotate_pose_inplane(cam1_T_w, rot1)
                    cam1_T_cam0 = cam1_T_w @ np.linalg.inv(cam0_T_w)
                    T_0to1 = cam1_T_cam0

                # 极线误差 → Precision
                epi_errs = compute_epipolar_error(mkpts0, mkpts1, T_0to1, K0, K1)
                correct = epi_errs < 5e-4
                num_correct = np.sum(correct)
                precision = np.mean(correct) if len(correct) > 0 else 0
                matching_score = num_correct / len(kpts0) if len(kpts0) > 0 else 0

                # RANSAC 姿态估计
                thresh = 1.
                ret = estimate_pose(mkpts0, mkpts1, K0, K1, thresh)
                if ret is None:
                    err_t, err_R = np.inf, np.inf
                else:
                    R, t, inliers = ret
                    err_t, err_R = compute_pose_error(T_0to1, R, t)

                # === Dense-Recall@τ（可选） ===
                dense_recall = None
                if opt.dense_eval:
                    scene_dir = pf.parent  # 当前 pairs.txt 所在目录
                    gt0, gt1 = None, None
                    if opt.dense_mode == 'npz':
                        gt0, gt1 = try_dense_from_npz(scene_dir, stem0, stem1)
                    elif opt.dense_mode == 'module':
                        gt0, gt1 = try_dense_from_module(scene_dir, None, name0, name1, image0, image1)
                    if gt0 is not None and gt1 is not None and len(gt0) > 0:
                        dense_recall = compute_dense_recall(mkpts0, mkpts1, gt0, gt1, opt.dense_thresh)
                    else:
                        print('[Dense] 未获得真值投影点或为空，跳过 Dense-Recall')

                out_eval = {
                    'error_t': err_t,
                    'error_R': err_R,
                    'precision': precision,
                    'matching_score': matching_score,
                    'num_correct': num_correct,
                    'epipolar_errors': epi_errs
                }
                if dense_recall is not None:
                    out_eval['dense_recall_tau'] = float(opt.dense_thresh)
                    out_eval['dense_recall'] = float(dense_recall)
                np.savez(str(eval_path), **out_eval)

            # ========== 匹配可视化 ==========
            if do_viz:
                color = cm.jet(mconf)  # 匹配点颜色由置信度决定
                text = [
                    'SuperGlue',
                    f'Keypoints: {len(kpts0)}:{len(kpts1)}',
                    f'Matches: {len(mkpts0)}'
                ]
                if rot0 != 0 or rot1 != 0:
                    text.append(f'Rotation: {rot0}:{rot1}')
                small_text = [
                    f'Keypoint Threshold: {matching.superpoint.config["keypoint_threshold"]:.4f}',
                    f'Match Threshold: {matching.superglue.config["match_threshold"]:.2f}',
                    f'Image Pair: {stem0}:{stem1}'
                ]
                make_matching_plot(
                    image0, image1, kpts0, kpts1, mkpts0, mkpts1, color,
                    text, viz_path, opt.show_keypoints,
                    opt.fast_viz, opt.opencv_display, 'Matches', small_text)

            # 打印计时器信息
            timer.print(f'Finished pair {i+1:5} of {len(pairs):5}')

        # ========== 汇总评估结果（含 Dense-Recall） ==========
        if opt.eval:
            pose_errors = []
            precisions = []
            dense_recalls = []
            matching_scores = []
            for pair in pairs:
                name0, name1 = pair[:2]
                stem0, stem1 = Path(name0).stem, Path(name1).stem
                eval_path = output_dir / f"{stem0}_{stem1}_evaluation.npz"
                results = np.load(eval_path)
                pose_error = np.maximum(results['error_t'], results['error_R'])
                pose_errors.append(pose_error)
                precisions.append(results['precision'])
                matching_scores.append(results['matching_score'])
                if 'dense_recall' in results:
                    dense_recalls.append(results['dense_recall'])
            thresholds = [5, 10, 20]
            aucs = pose_auc(pose_errors, thresholds)
            aucs = [100.*yy for yy in aucs]
            prec = 100.*np.mean(precisions)
            ms = 100.*np.mean(matching_scores)
            print('Evaluation Results (mean over {} pairs):'.format(len(pairs)))
            print('AUC@5	 AUC@10	 AUC@20	 Prec	 MScore	 Dense-Recall@{}px'.format(opt.dense_thresh))
            dr = 100.*np.mean(dense_recalls) if len(dense_recalls) > 0 else float('nan')
            print('{:.2f}	 {:.2f}	 {:.2f}	 {:.2f}	 {:.2f}	 {:.2f}'.format(
                aucs[0], aucs[1], aucs[2], prec, ms, dr))

        print(f"✅ 完成 {pf}")
