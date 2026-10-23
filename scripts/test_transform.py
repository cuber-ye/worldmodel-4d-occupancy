"""
验证 transform.py 的正确性（对齐 transform.py 实际 API）
"""

import numpy as np
from transform import (
    euler_to_rotation,
    construct_extrinsic,
    _axis_transform,
    _pose_from_nuscenes,
    project_points,
    LidarToCamera,
)


def test_rotation_properties():
    print("测试1: 旋转矩阵性质...")
    R = euler_to_rotation(0.1, 0.2, 0.3)
    assert R.shape == (3, 3)
    assert np.allclose(R @ R.T, np.eye(3)), "R @ R.T 应等于 I"
    assert np.allclose(np.linalg.inv(R), R.T), "R^(-1) 应等于 R^T"
    assert np.allclose(np.linalg.det(R), 1.0), "det(R) 应等于 1"
    print("  ✅ 通过")


def test_extrinsic_shape():
    print("测试2: 外参矩阵形状...")
    T = construct_extrinsic((0.1, 0.2, 0.3), [1.0, 2.0, 3.0])
    assert T.shape == (4, 4), f"T 应为 (4,4), 实际 {T.shape}"
    assert np.allclose(T[3, :], [0, 0, 0, 1]), "最后一行应为 [0,0,0,1]"
    print("  ✅ 通过")


def test_axis_transform():
    print("测试3: nuScenes 轴变换矩阵...")
    M = _axis_transform()
    assert M.shape == (3, 3)
    # 标准 nuScenes 轴变换: 正交矩阵
    assert np.allclose(M @ M.T, np.eye(3)), "轴变换应为正交矩阵"
    # cam_x = -lidar_y → 第一行 [0,-1,0]
    assert np.allclose(M[0, :], [0, -1, 0])
    # cam_z = lidar_x → 第三行 [1,0,0]
    assert np.allclose(M[2, :], [1, 0, 0])
    print("  ✅ 通过")


def test_pose_from_nuscenes():
    print("测试4: 四元数 → 变换矩阵...")
    # 单位四元数 = 无旋转
    T = _pose_from_nuscenes([1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
    assert T.shape == (4, 4)
    assert np.allclose(T[:3, :3], np.eye(3)), "单位四元数应得单位旋转"
    assert np.allclose(T[:3, 3], [0, 0, 0])
    assert np.allclose(T[3, :], [0, 0, 0, 1])
    print("  ✅ 通过")


def test_optical_center():
    print("测试5: 光心投影验证...")
    K = np.array([[1266.4, 0, 800], [0, 1266.4, 450], [0, 0, 1]])
    cx, cy = K[0, 2], K[1, 2]

    # 构造: 相机前方 10m 正中央的点
    # camera 系下 (0, 0, 10) → 反算回 lidar 系
    R = euler_to_rotation(0.01, -0.02, 0.005) @ _axis_transform()
    t = np.array([1.5, 0.0, 1.6]).reshape(3, 1)
    T = np.vstack([np.hstack([R, t]), np.array([[0, 0, 0, 1]])])

    point_cam = np.array([0.0, 0.0, 10.0, 1.0])
    point_lidar = np.linalg.inv(T) @ point_cam
    point_lidar = point_lidar[:3]

    proj = LidarToCamera(K, T, 1600, 900)
    pixels, depth, mask = proj.project(
        point_lidar.reshape(1, 3), return_depth=True
    )

    print(f"  前方正中央的点投影到: ({pixels[0, 0]:.1f}, {pixels[0, 1]:.1f})")
    print(f"  光心: ({cx}, {cy})")
    assert abs(pixels[0, 0] - cx) < 5
    assert abs(pixels[0, 1] - cy) < 5
    print("  ✅ 通过")


def test_batch_dimensions():
    print("测试6: 批量维度 & 过滤...")
    K = np.array([[1266.4, 0, 800], [0, 1266.4, 450], [0, 0, 1]])
    R = euler_to_rotation(0.01, -0.02, 0.005) @ _axis_transform()
    t = np.array([1.5, 0.0, 1.6]).reshape(3, 1)
    T = np.vstack([np.hstack([R, t]), np.array([[0, 0, 0, 1]])])

    proj = LidarToCamera(K, T, 1600, 900)

    # (N, 3)
    pts = np.random.default_rng(0).normal(
        size=(100, 3)
    ) * 3 + np.array([15, 0, 0])
    pixels, depth, mask = proj.project(pts, return_depth=True)
    assert pixels.shape[1] == 2
    assert depth.shape[0] == pixels.shape[0] == mask.sum()
    assert mask.shape == (100,)
    print(f"  输入100点 → 有效 {pixels.shape[0]} 点, "
          f"深度 [{depth.min():.1f}, {depth.max():.1f}] m")
    print("  ✅ 通过")


def test_occluded_filter():
    print("测试7: 后方点被过滤...")
    K = np.array([[1266.4, 0, 800], [0, 1266.4, 450], [0, 0, 1]])
    R = euler_to_rotation(0.01, -0.02, 0.005) @ _axis_transform()
    t = np.array([1.5, 0.0, 1.6]).reshape(3, 1)
    T = np.vstack([np.hstack([R, t]), np.array([[0, 0, 0, 1]])])
    proj = LidarToCamera(K, T, 1600, 900)

    # 相机后方 + 前方 混合
    pts = np.array([
        [10.0, 0.0, -1.2],    # 前方
        [-5.0, 0.0, 0.0],     # 后方（lidar 坐标系后方）
        [20.0, 3.0, 0.5],     # 前方
    ])
    pixels, depth, mask = proj.project(pts, return_depth=True)
    print(f"  3个点中 {mask.sum()} 个有效（后方点被过滤）")
    assert mask.sum() < 3, "至少应有1个被过滤"
    print("  ✅ 通过")


def test_info():
    print("测试8: info() 返回关键参数...")
    K = np.array([[1266.4, 0, 800], [0, 1266.4, 450], [0, 0, 1]])
    T = construct_extrinsic((0.1, 0.2, 0.3), [1.0, 2.0, 3.0])
    proj = LidarToCamera(K, T, 1600, 900)
    info = proj.info()
    for key in ["K", "R", "t", "img_size", "fx", "fy", "cx", "cy"]:
        assert key in info, f"缺少 {key}"
    print(f"  fx={info['fx']}, cx={info['cx']}, img={info['img_size']}")
    print("  ✅ 通过")


def test_project_points_one_liner():
    print("测试9: 便捷函数 project_points()...")
    K = np.array([[1266.4, 0, 800], [0, 1266.4, 450], [0, 0, 1]])
    T = construct_extrinsic((0.01, -0.02, 0.005), [1.5, 0.0, 1.6])
    pts = np.array([[10.0, 0.0, -1.2], [15.0, 2.0, 0.5]])
    pixels = project_points(pts, K, T, 1600, 900)
    assert pixels.ndim == 2 and pixels.shape[1] == 2
    print(f"  一行调用 → {pixels.shape}")
    print("  ✅ 通过")


if __name__ == "__main__":
    print("=" * 50)
    print("  transform.py 验证测试（对齐实际 API）")
    print("=" * 50)

    test_rotation_properties()
    test_extrinsic_shape()
    test_axis_transform()
    test_pose_from_nuscenes()
    test_optical_center()
    test_batch_dimensions()
    test_occluded_filter()
    test_info()
    test_project_points_one_liner()

    print("=" * 50)
    print("  ✅ 全部测试通过！")
    print("=" * 50)