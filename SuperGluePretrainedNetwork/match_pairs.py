from pathlib import Path
import argparse
import random
import numpy as np
import matplotlib.cm as cm
import torch

# 导入SuperGlue相关模块
from models.matching import Matching  # SuperPoint+SuperGlue主模型
from models.utils import (
    compute_pose_error,         # 计算姿态误差
    compute_epipolar_error,     # 计算极线误差
    estimate_pose,              # 用匹配点估计相机姿态
    make_matching_plot,         # 绘制匹配可视化
    error_colormap,             # 错误可视化配色
    AverageTimer,               # 计时工具
    pose_auc,                   # 姿态AUC评估
    read_image,                 # 读取并预处理图像
    rotate_intrinsics,          # 旋转相机内参
    rotate_pose_inplane,        # 旋转相机外参
    scale_intrinsics            # 缩放相机内参
)

torch.set_grad_enabled(False)  # 关闭梯度计算，加速推理

if __name__ == '__main__':
    # ========== 命令行参数解析 ==========
    parser = argparse.ArgumentParser(
        description='Image pair matching and pose evaluation with SuperGlue',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # 输入：图像对列表文件（每一行写两张图像路径）
    parser.add_argument(
        '--input_pairs', type=str,
        default='navi_v1.0/duck_bath_yellow_s/multiview-04-pixel_6pro/pairs.txt',
        help='Path to the list of image pairs')
    parser.add_argument(
        '--input_dir', type=str, default='.',   # 不再使用
        help='(Unused, since pairs.txt already has relative paths)')

    # 输出：匹配结果和可视化保存位置 不再使用
    parser.add_argument(
        '--output_dir', type=str, default='dump_match_pairs/',
        help='Path to the directory in which the .npz results and optionally,'
             'the visualization images are written')
    # 最大处理多少对
    parser.add_argument(
        '--max_length', type=int, default=-1,
        help='Maximum number of pairs to evaluate')
    # 输入图像缩放
    parser.add_argument(
        '--resize', type=int, nargs='+', default=[640, 480],
        help='Resize the input image before running inference. If two numbers, '
             'resize to the exact dimensions, if one number, resize the max '
             'dimension, if -1, do not resize')
    parser.add_argument(
        '--resize_float', action='store_true',
        help='Resize the image after casting uint8 to float')
    # SuperGlue 配置
    parser.add_argument('--superglue', choices={'indoor', 'outdoor'}, default='indoor',
        help='SuperGlue weights')
    parser.add_argument('--max_keypoints', type=int, default=1024,
        help='Max number of SuperPoint关键点')
    parser.add_argument('--keypoint_threshold', type=float, default=0.005,
        help='SuperPoint检测阈值')
    parser.add_argument('--nms_radius', type=int, default=4,
        help='SuperPoint非极大抑制半径')
    parser.add_argument('--sinkhorn_iterations', type=int, default=20,
        help='SuperGlue内部Sinkhorn迭代次数')
    parser.add_argument('--match_threshold', type=float, default=0.2,
        help='SuperGlue匹配分数阈值')
    # 可视化和评估相关参数
    parser.add_argument('--viz', action='store_true',
        help='是否保存匹配可视化结果')
    parser.add_argument('--eval', action='store_true',
        help='是否进行姿态估计评估 (需要GT位姿)')
    parser.add_argument('--fast_viz', action='store_true',
        help='是否用OpenCV快速画图')
    parser.add_argument('--cache', action='store_true',
        help='若已有结果文件，是否跳过计算')
    parser.add_argument('--show_keypoints', action='store_true',
        help='可视化关键点')
    parser.add_argument('--viz_extension', type=str, default='png', choices=['png', 'pdf'],
        help='可视化结果文件格式')
    parser.add_argument('--opencv_display', action='store_true',
        help='用OpenCV实时显示匹配结果')
    parser.add_argument('--shuffle', action='store_true',
        help='随机打乱输入对顺序')
    parser.add_argument('--force_cpu', action='store_true',
        help='强制使用CPU')

    opt = parser.parse_args()
    print(opt)
    # === 手动强制可视化与评估 ===
    opt.viz = True        # 始终保存匹配可视化
    #opt.eval = True       # 始终做姿态评估（需要 pairs.txt 有GT）
    # opt.fast_viz = True # 如果想强制用OpenCV快速画图，可以加上
    # opt.cache = False   # 如果想每次都重新画图，关掉缓存
    print("⚙️ 已强制启用可视化和评估模式")
    # ========== 参数合法性检查 ==========
    # 检查可视化参数组合是否合法
    assert not (opt.opencv_display and not opt.viz), 'Must use --viz with --opencv_display'
    assert not (opt.opencv_display and not opt.fast_viz), 'Cannot use --opencv_display without --fast_viz'
    assert not (opt.fast_viz and not opt.viz), 'Must use --viz with --fast_viz'
    assert not (opt.fast_viz and opt.viz_extension == 'pdf'), 'Cannot use pdf extension with --fast_viz'

    # ========== 图像缩放参数处理 ==========
    # 处理resize参数，支持多种缩放方式
    if len(opt.resize) == 2 and opt.resize[1] == -1:
        opt.resize = opt.resize[0:1]
    if len(opt.resize) == 2:
        print('Will resize to {}x{} (WxH)'.format(
            opt.resize[0], opt.resize[1]))
    elif len(opt.resize) == 1 and opt.resize[0] > 0:
        print('Will resize max dimension to {}'.format(opt.resize[0]))
    elif len(opt.resize) == 1:
        print('Will not resize images')
    else:
        raise ValueError('Cannot specify more than two integers for --resize')

    # ========== 读取图像对列表 ==========
    pairs = []
    with open(opt.input_pairs, 'r') as f:
        for line in f:
            if line.strip() == "":
                continue
            # 只取前两个字段作为图像名，后面可能有GT参数
            name0, name1 = line.strip().split()[:2]
            pairs.append(line.strip().split())

    # ========== 限制最大处理对数 ==========
    if opt.max_length > -1:
        pairs = pairs[0:np.min([len(pairs), opt.max_length])]

    # ========== 随机打乱对顺序 ==========
    if opt.shuffle:
        random.Random(0).shuffle(pairs)

    # ========== 检查评估模式下的输入格式 ==========
    # 如果需要评估，确保每对都包含38个字段（图像名+旋转+内参+外参）
    if opt.eval:
        if not all([len(p) == 38 for p in pairs]):
            raise ValueError(
                'All pairs should have ground truth info for evaluation.'
                'File \"{}\" needs 38 valid entries per row'.format(opt.input_pairs))

    # ========== 加载SuperPoint和SuperGlue模型 ==========
    device = 'cuda' if torch.cuda.is_available() and not opt.force_cpu else 'cpu'
    print('Running inference on device \"{}\"'.format(device))
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
    matching = Matching(config).eval().to(device)  # 加载模型到指定设备

    # ========== 创建输出目录 ==========
    input_dir = Path(opt.input_dir)
    print('Looking for data in directory \"{}\"'.format(input_dir))
    # output_dir = Path(opt.output_dir)
    # output_dir.mkdir(exist_ok=True, parents=True)
    # print('Will write matches to directory \"{}\"'.format(output_dir))

    # 自动把输出放到 pairs.txt 的同级目录
    pairs_path = Path(opt.input_pairs)
    output_dir = pairs_path.parent / "dump_match_pairs"
    output_dir.mkdir(exist_ok=True, parents=True)
    print(f"Will write matches to directory \"{output_dir}\"")
    if opt.eval:
        print('Will write evaluation results',
              'to directory \"{}\"'.format(output_dir))
    if opt.viz:
        print('Will write visualization images to',
              'directory \"{}\"'.format(output_dir))

    timer = AverageTimer(newline=True)  # 初始化计时器

    # ========== 主循环：遍历每一对图像 ==========
    for i, pair in enumerate(pairs):
        # 取出图像名
        name0, name1 = pair[:2]
        stem0, stem1 = Path(name0).stem, Path(name1).stem
        # 构造输出文件路径
        matches_path = output_dir / '{}_{}_matches.npz'.format(stem0, stem1)
        eval_path = output_dir / '{}_{}_evaluation.npz'.format(stem0, stem1)
        viz_path = output_dir / '{}_{}_matches.{}'.format(stem0, stem1, opt.viz_extension)
        viz_eval_path = output_dir / '{}_{}_evaluation.{}'.format(stem0, stem1, opt.viz_extension)

        # ========== 缓存机制：若已有结果则跳过 ==========
        do_match = True
        do_eval = opt.eval
        do_viz = opt.viz
        do_viz_eval = opt.eval and opt.viz
        if opt.cache:
            # 匹配结果缓存
            if matches_path.exists():
                try:
                    results = np.load(matches_path)
                except:
                    raise IOError('Cannot load matches .npz file: %s' % matches_path)
                kpts0, kpts1 = results['keypoints0'], results['keypoints1']
                matches, conf = results['matches'], results['match_confidence']
                do_match = False
            # 评估结果缓存
            if opt.eval and eval_path.exists():
                try:
                    results = np.load(eval_path)
                except:
                    raise IOError('Cannot load eval .npz file: %s' % eval_path)
                err_R, err_t = results['error_R'], results['error_t']
                precision = results['precision']
                matching_score = results['matching_score']
                num_correct = results['num_correct']
                epi_errs = results['epipolar_errors']
                do_eval = False
            # 匹配可视化缓存
            if opt.viz and viz_path.exists():
                do_viz = False
            # 评估可视化缓存
            if opt.viz and opt.eval and viz_eval_path.exists():
                do_viz_eval = False
            timer.update('load_cache')

        # 如果所有输出都已存在，直接跳过
        if not (do_match or do_eval or do_viz or do_viz_eval):
            timer.print('Finished pair {:5} of {:5}'.format(i, len(pairs)))
            continue

        # ========== 处理EXIF旋转信息 ==========
        # 若有旋转信息，取出，否则默认为0
        if len(pair) >= 5:
            rot0, rot1 = int(pair[2]), int(pair[3])
        else:
            rot0, rot1 = 0, 0

        # ========== 读取图像 ==========
        # 返回原图、归一化张量、缩放比例
        image0, inp0, scales0 = read_image(
            Path(name0), device, opt.resize, rot0, opt.resize_float)
        image1, inp1, scales1 = read_image(
            Path(name1), device, opt.resize, rot1, opt.resize_float)


        if image0 is None or image1 is None:
            print('Problem reading image pair: {} {}'.format(
                input_dir/name0, input_dir/name1))
            exit(1)
        timer.update('load_image')

        # ========== 关键点匹配 ==========
        if do_match:
            # 执行SuperGlue匹配，返回关键点、匹配索引、置信度等
            pred = matching({'image0': inp0, 'image1': inp1})
            pred = {k: v[0].cpu().numpy() for k, v in pred.items()}
            kpts0, kpts1 = pred['keypoints0'], pred['keypoints1']
            matches, conf = pred['matches0'], pred['matching_scores0']
            timer.update('matcher')

            # 保存匹配结果到npz
            out_matches = {'keypoints0': kpts0, 'keypoints1': kpts1,
                           'matches': matches, 'match_confidence': conf}
            np.savez(str(matches_path), **out_matches)

        # ========== 提取有效匹配点 ==========
        # matches为每个kpts0的匹配索引，-1表示未匹配
        valid = matches > -1
        mkpts0 = kpts0[valid]              # 有效的关键点0
        mkpts1 = kpts1[matches[valid]]     # 匹配到的关键点1
        mconf = conf[valid]                # 匹配置信度

        # ========== 姿态估计与评估 ==========
        if do_eval:
            # 读取GT内参、外参
            assert len(pair) == 38, 'Pair does not have ground truth info'
            K0 = np.array(pair[4:13]).astype(float).reshape(3, 3)
            K1 = np.array(pair[13:22]).astype(float).reshape(3, 3)
            T_0to1 = np.array(pair[22:]).astype(float).reshape(4, 4)

            # 缩放内参以适配图像resize
            K0 = scale_intrinsics(K0, scales0)
            K1 = scale_intrinsics(K1, scales1)

            # 若有旋转，需同步旋转内参和外参
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

            # 计算极线误差（用于评估匹配精度）
            epi_errs = compute_epipolar_error(mkpts0, mkpts1, T_0to1, K0, K1)
            correct = epi_errs < 5e-4
            num_correct = np.sum(correct)
            precision = np.mean(correct) if len(correct) > 0 else 0
            matching_score = num_correct / len(kpts0) if len(kpts0) > 0 else 0

            # 用RANSAC等方法估计相对姿态
            thresh = 1.  # 单位像素
            ret = estimate_pose(mkpts0, mkpts1, K0, K1, thresh)
            if ret is None:
                err_t, err_R = np.inf, np.inf
            else:
                R, t, inliers = ret
                err_t, err_R = compute_pose_error(T_0to1, R, t)

            # 保存评估结果
            out_eval = {'error_t': err_t,
                        'error_R': err_R,
                        'precision': precision,
                        'matching_score': matching_score,
                        'num_correct': num_correct,
                        'epipolar_errors': epi_errs}
            np.savez(str(eval_path), **out_eval)
            timer.update('eval')

        # ========== 匹配可视化 ==========
        if do_viz:
            # 画出匹配点，颜色根据置信度
            color = cm.jet(mconf)
            text = [
                'SuperGlue',
                'Keypoints: {}:{}'.format(len(kpts0), len(kpts1)),
                'Matches: {}'.format(len(mkpts0)),
            ]
            if rot0 != 0 or rot1 != 0:
                text.append('Rotation: {}:{}'.format(rot0, rot1))

            # 显示参数信息
            k_thresh = matching.superpoint.config['keypoint_threshold']
            m_thresh = matching.superglue.config['match_threshold']
            small_text = [
                'Keypoint Threshold: {:.4f}'.format(k_thresh),
                'Match Threshold: {:.2f}'.format(m_thresh),
                'Image Pair: {}:{}'.format(stem0, stem1),
            ]

            make_matching_plot(
                image0, image1, kpts0, kpts1, mkpts0, mkpts1, color,
                text, viz_path, opt.show_keypoints,
                opt.fast_viz, opt.opencv_display, 'Matches', small_text)

            timer.update('viz_match')

        # ========== 评估可视化 ==========
        if do_viz_eval:
            # 画出姿态评估结果，颜色根据极线误差
            color = np.clip((epi_errs - 0) / (1e-3 - 0), 0, 1)
            color = error_colormap(1 - color)
            deg, delta = ' deg', 'Delta '
            if not opt.fast_viz:
                deg, delta = '°', '$\\Delta$'
            e_t = 'FAIL' if np.isinf(err_t) else '{:.1f}{}'.format(err_t, deg)
            e_R = 'FAIL' if np.isinf(err_R) else '{:.1f}{}'.format(err_R, deg)
            text = [
                'SuperGlue',
                '{}R: {}'.format(delta, e_R), '{}t: {}'.format(delta, e_t),
                'inliers: {}/{}'.format(num_correct, (matches > -1).sum()),
            ]
            if rot0 != 0 or rot1 != 0:
                text.append('Rotation: {}:{}'.format(rot0, rot1))

            # 显示参数信息
            k_thresh = matching.superpoint.config['keypoint_threshold']
            m_thresh = matching.superglue.config['match_threshold']
            small_text = [
                'Keypoint Threshold: {:.4f}'.format(k_thresh),
                'Match Threshold: {:.2f}'.format(m_thresh),
                'Image Pair: {}:{}'.format(stem0, stem1),
            ]

            make_matching_plot(
                image0, image1, kpts0, kpts1, mkpts0,
                mkpts1, color, viz_eval_path,
                text, opt.show_keypoints, opt.fast_viz,
                opt.opencv_display, 'Relative Pose', small_text)

            timer.update('viz_eval')

        timer.print('Finished pair {:5} of {:5}'.format(i, len(pairs)))

    #========== 汇总评估结果（已注释） ==========
    if opt.eval:
        # 汇总所有pair的评估结果，计算AUC等指标
        pose_errors = []
        precisions = []
        matching_scores = []
        for pair in pairs:
            name0, name1 = pair[:2]
            stem0, stem1 = Path(name0).stem, Path(name1).stem
            eval_path = output_dir / \
                '{}_{}_evaluation.npz'.format(stem0, stem1)
            results = np.load(eval_path)
            pose_error = np.maximum(results['error_t'], results['error_R'])
            pose_errors.append(pose_error)
            precisions.append(results['precision'])
            matching_scores.append(results['matching_score'])
        thresholds = [5, 10, 20]
        aucs = pose_auc(pose_errors, thresholds)
        aucs = [100.*yy for yy in aucs]
        prec = 100.*np.mean(precisions)
        ms = 100.*np.mean(matching_scores)
        print('Evaluation Results (mean over {} pairs):'.format(len(pairs)))
        print('AUC@5\t AUC@10\t AUC@20\t Prec\t MScore\t')
        print('{:.2f}\t {:.2f}\t {:.2f}\t {:.2f}\t {:.2f}\t'.format(
            aucs[0], aucs[1], aucs[2], prec, ms))