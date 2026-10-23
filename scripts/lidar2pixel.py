"""
第1-3步完整脚本：激光雷达3D点 → 图像像素坐标

链路：
  第②步 外参变换: lidar 3D → camera 3D
  第③步 内参投影: camera 3D → pixel 2D
  过滤: 深度>0 + 图像范围内

坐标系约定（nuScenes）：
  lidar/ego: x=前, y=左, z=上 (右手系, y朝左)
  camera:    x=右, y=下, z=前 (右手系)
  轴变换: cam_x=-ego_y, cam_y=-ego_z, cam_z=ego_x
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# 工具函数
# ============================================================

def axis_transform_matrix():
    """
    nuScenes lidar/ego → camera 坐标轴变换
    不含传感器安装位姿, 仅做坐标轴对齐
    """
    return np.array([
        [ 0, -1,  0],   # cam_x = -lidar_y
        [ 0,  0, -1],   # cam_y = -lidar_z
        [ 1,  0,  0],   # cam_z =  lidar_x
    ])


def euler_to_rotation(yaw, pitch, roll):
    """
    欧拉角 → 旋转矩阵 (ZYX 顺序)
    """
    Rz = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw),  np.cos(yaw), 0],
        [0,            0,           1]
    ])
    Ry = np.array([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0,             1, 0           ],
        [-np.sin(pitch),0, np.cos(pitch)]
    ])
    Rx = np.array([
        [1, 0,             0           ],
        [0, np.cos(roll),  -np.sin(roll)],
        [0, np.sin(roll),   np.cos(roll)]
    ])
    return Rz @ Ry @ Rx


def construct_extrinsic(rotation_euler, translation):
    """
    构造 4×4 外参矩阵 T = [R | t]
    """
    R = euler_to_rotation(*rotation_euler)
    t = np.array(translation).reshape(3, 1)
    T = np.vstack([
        np.hstack([R, t]),
        np.array([0, 0, 0, 1])
    ])
    return T


def build_calibration():
    """
    构造模拟标定参数（对应 nuScenes 前视相机）

    返回:
        K: 3×3 内参矩阵
        T_lidar2cam: 4×4 外参矩阵（lidar → camera）
    """
    # ---- 内参 ----
    fx, fy = 1266.4, 1266.4
    cx, cy = 816.3, 491.3
    K = np.array([
        [fx,  0,  cx],
        [0,  fy,  cy],
        [0,   0,   1]
    ])

    # ---- 外参 ----
    # 传感器安装位姿的旋转部分（小角度近似无旋转）
    R_install = euler_to_rotation(0.01, -0.02, 0.005)

    # 完整旋转 = 安装旋转 @ 轴变换
    R_axis = axis_transform_matrix()
    R = R_install @ R_axis

    # 平移: 相机相对 lidar 的位置（前方1.5m, 上方1.6m）
    t = np.array([1.5, 0.0, 1.6]).reshape(3, 1)

    T_lidar2cam = np.vstack([
        np.hstack([R, t]),
        np.array([0, 0, 0, 1])
    ])

    return K, T_lidar2cam


# ============================================================
# 核心函数
# ============================================================

def lidar_to_camera(points_lidar, T_lidar2cam):
    """
    第②步：外参变换
    输入: (N, 3) lidar 坐标系下的3D点
    输出: (N, 3) camera 坐标系下的3D点
    """
    N = len(points_lidar)
    # 齐次坐标: (N, 4)
    points_hom = np.hstack([points_lidar, np.ones((N, 1))])
    # 变换: (N, 4) @ (4, 4).T → (N, 4)
    points_cam = points_hom @ T_lidar2cam.T
    return points_cam[:, :3]


def camera_to_pixel(points_cam, K):
    """
    第③步：内参投影
    输入: (N, 3) camera 坐标系下的3D点
    输出: pixels (N, 2), depth (N,)
    """
    # 矩阵乘法: (N, 3) @ (3, 3).T → (N, 3)
    points_hom = points_cam @ K.T
    # 透视除法
    Zc = points_hom[:, 2]
    u = points_hom[:, 0] / Zc
    v = points_hom[:, 1] / Zc
    return np.stack([u, v], axis=1), Zc


def lidar_to_pixel(points_lidar, K, T_lidar2cam, img_w, img_h):
    """
    完整链路 + 过滤
    """
    # 第②步
    points_cam = lidar_to_camera(points_lidar, T_lidar2cam)

    # 第③步
    pixels, depth = camera_to_pixel(points_cam, K)

    # 过滤
    valid = (
        (depth > 0) &
        (pixels[:, 0] >= 0) & (pixels[:, 0] < img_w) &
        (pixels[:, 1] >= 0) & (pixels[:, 1] < img_h)
    )

    return pixels[valid], depth[valid], points_cam[valid], valid


# ============================================================
# 演示
# ============================================================

def main():
    print("=" * 60)
    print("  第1-3步完整演示：激光雷达 → 像素坐标")
    print("=" * 60)

    IMG_W, IMG_H = 1600, 900

    # 1. 构建标定参数
    K, T_lidar2cam = build_calibration()
    print("\n【内参矩阵 K】")
    print(K)
    print("\n【外参矩阵 T_lidar2cam】")
    print(T_lidar2cam)

    # 2. 构造模拟 lidar 点云
    np.random.seed(42)
    N_ground = 500
    ground_x = np.random.uniform(1, 40, N_ground)
    ground_y = np.random.uniform(-15, 15, N_ground)
    ground_z = np.full(N_ground, -1.2)

    # 前方几个"障碍物"
    obs = np.array([
        [8.0,  -2.0, 0.5],
        [8.0,  -1.5, 0.5],
        [8.0,  -1.0, 0.5],
        [8.0,  -1.0, 1.0],
        [8.0,  -1.0, 1.5],
        [15.0,  3.0, 0.5],
        [15.0,  3.5, 0.5],
        [15.0,  3.0, 1.0],
        [20.0,  0.0, 0.5],
        [20.0,  0.0, 1.0],
        [20.0,  0.5, 1.0],
        [20.0, -0.5, 1.0],
    ])

    points_lidar = np.vstack([
        np.column_stack([ground_x, ground_y, ground_z]),
        obs
    ])

    print(f"\n【输入】lidar 点云: {len(points_lidar)} 个点")

    # 3. 完整变换
    pixels, depth, points_cam, valid_mask = lidar_to_pixel(
        points_lidar, K, T_lidar2cam, IMG_W, IMG_H
    )

    print(f"\n【变换后】{len(pixels)} 个点投影到图像范围内（共 {len(points_lidar)} 个点）")
    if len(pixels) > 0:
        print(f"  像素 u 范围: [{pixels[:, 0].min():.1f}, {pixels[:, 0].max():.1f}]")
        print(f"  像素 v 范围: [{pixels[:, 1].min():.1f}, {pixels[:, 1].max():.1f}]")
        print(f"  深度 Zc 范围: [{depth.min():.1f}, {depth.max():.1f}]")

    # 4. 逐步验证
    print("\n" + "-" * 60)
    print("【逐步验证】手动算第一个点:")
    p0 = points_lidar[0]
    print(f"  原始 lidar 坐标: ({p0[0]:.2f}, {p0[1]:.2f}, {p0[2]:.2f})")

    cam0 = lidar_to_camera(p0.reshape(1, 3), T_lidar2cam)[0]
    print(f"  第②步 外参变换 → camera: ({cam0[0]:.2f}, {cam0[1]:.2f}, {cam0[2]:.2f})")

    pix0, z0 = camera_to_pixel(cam0.reshape(1, 3), K)
    print(f"  第③步 内参投影 → pixel: ({pix0[0, 0]:.1f}, {pix0[0, 1]:.1f}), 深度={z0[0]:.1f}")

    # 5. 可视化
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    ax = axes[0]
    ax.scatter(points_lidar[:, 0], points_lidar[:, 1],
              c="#37d6e6", s=3, alpha=0.4, label="地面点")
    ax.scatter(obs[:, 0], obs[:, 1], c="#ff7c8a", s=30, marker="^", label="障碍物")
    ax.set_xlabel("X (前方, m)")
    ax.set_ylabel("Y (左右, m)")
    ax.set_title("BEV 视角 (Lidar坐标系)")
    ax.legend()
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.invert_yaxis()

    ax = axes[1]
    ax.imshow(np.zeros((IMG_H, IMG_W, 3)), cmap="gray",
              extent=[0, IMG_W, IMG_H, 0])

    valid_ground = valid_mask[:N_ground]
    ground_pixels = pixels[:np.sum(valid_ground)]
    ax.scatter(ground_pixels[:, 0], ground_pixels[:, 1],
              c="#37d6e6", s=1, alpha=0.3, label="地面点投影")

    obs_pixels = pixels[np.sum(valid_ground):]
    if len(obs_pixels) > 0:
        ax.scatter(obs_pixels[:, 0], obs_pixels[:, 1],
                  c="#ff7c8a", s=20, marker="^", label="障碍物投影")

    ax.set_xlim(0, IMG_W)
    ax.set_ylim(IMG_H, 0)
    ax.set_xlabel("u (像素)")
    ax.set_ylabel("v (像素)")
    ax.set_title("图像平面投影结果")
    ax.legend()

    plt.tight_layout()
    plt.savefig("lidar_projection_result.png", dpi=150, bbox_inches="tight")
    print("\n✅ 可视化已保存: lidar_projection_result.png")

    print("\n" + "=" * 60)
    print("  全部完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()