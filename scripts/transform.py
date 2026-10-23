"""
坐标转换模块：激光雷达点云 → 相机像素坐标

功能：
  - 外参变换 (lidar/ego → camera)
  - 内参投影 (camera → pixel)
  - 边界过滤
  - 可视化验证
  - CLI 接口

用法:
  作为模块:
    from transform import LidarToCamera, project_points

  命令行:
    python transform.py --num-points 2000 --visualize
"""

import os
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# 核心类
# ============================================================

class LidarToCamera:
    """
    激光雷达到相机的投影器

    封装内参 K、外参 T_lidar2cam、图像尺寸，
    一行调用完成 lidar → pixel 的全链路投影。
    """

    def __init__(self, K, T_lidar2cam, img_w, img_h):
        """
        K:            (3, 3) 内参矩阵
        T_lidar2cam:  (4, 4) 外参矩阵 (lidar → camera)
        img_w:        图像宽度 (像素)
        img_h:        图像高度 (像素)
        """
        self.K = np.asarray(K, dtype=np.float64).reshape(3, 3)
        self.T = np.asarray(T_lidar2cam, dtype=np.float64).reshape(4, 4)
        self.img_w = int(img_w)
        self.img_h = int(img_h)

        self.R = self.T[:3, :3]
        self.t = self.T[:3, 3]

    @classmethod
    def from_nuscenes(cls, sensor_json, ego_pose_json=None,
                      img_w=1600, img_h=900):
        """
        从 nuScenes 的 calibrated_sensor.json 构造

        sensor_json:     dict, calibrated_sensor 记录
                         (含 'rotation' 四元数, 'translation', 'camera_intrinsic')
        ego_pose_json:   dict, ego_pose 记录（可选，用于转到全局坐标）
        img_w/img_h:     图像尺寸
        """
        K = np.array(sensor_json["camera_intrinsic"])

        # calibrated_sensor 给的是 sensor → ego 的变换
        # 转成 ego → sensor 需要取逆
        sensor2ego = _pose_from_nuscenes(
            sensor_json["rotation"],
            sensor_json["translation"]
        )
        ego2sensor = np.linalg.inv(sensor2ego)

        if ego_pose_json is not None:
            ego2global = _pose_from_nuscenes(
                ego_pose_json["rotation"],
                ego_pose_json["translation"]
            )
            T = ego2global @ ego2sensor
        else:
            T = ego2sensor

        return cls(K, T, img_w, img_h)

    # --------------------------------------------------------
    # 第②步：外参变换
    # --------------------------------------------------------
    def lidar_to_camera(self, points_lidar):
        """
        points_lidar: (N, 3) 或 (3, N)
        返回:           (N, 3) camera 坐标系下的点
        """
        pts = np.asarray(points_lidar)
        if pts.shape[0] == 3 and pts.shape[1] != 3:
            pts = pts.T
        N = len(pts)

        hom = np.hstack([pts, np.ones((N, 1))])   # (N, 4)
        cam = hom @ self.T.T                        # (N, 4)
        return cam[:, :3]                           # (N, 3)

    # --------------------------------------------------------
    # 第③步：内参投影
    # --------------------------------------------------------
    def camera_to_pixel(self, points_cam):
        """
        points_cam: (N, 3)
        返回:        pixels (N, 2), depth (N,)
        """
        proj = points_cam @ self.K.T                # (N, 3)
        Zc = proj[:, 2]
        u = proj[:, 0] / Zc
        v = proj[:, 1] / Zc
        return np.stack([u, v], axis=1), Zc

    # --------------------------------------------------------
    # 完整链路
    # --------------------------------------------------------
    def project(self, points_lidar, return_depth=False):
        """
        lidar 3D → pixel 2D 完整投影 + 过滤

        返回:
            默认:          (M, 2) 有效像素坐标
            return_depth:  (pixels (M,2), depth (M,), mask (N,))
        """
        points_cam = self.lidar_to_camera(points_lidar)
        pixels, depth = self.camera_to_pixel(points_cam)

        mask = self._valid_mask(pixels, depth)

        if return_depth:
            return pixels[mask], depth[mask], mask
        return pixels[mask]

    def _valid_mask(self, pixels, depth):
        return (
            (depth > 0) &
            (pixels[:, 0] >= 0) & (pixels[:, 0] < self.img_w) &
            (pixels[:, 1] >= 0) & (pixels[:, 1] < self.img_h)
        )

    # --------------------------------------------------------
    # 信息
    # --------------------------------------------------------
    def info(self):
        return {
            "K": self.K.tolist(),
            "R": self.R.tolist(),
            "t": self.t.tolist(),
            "img_size": (self.img_w, self.img_h),
            "fx": float(self.K[0, 0]),
            "fy": float(self.K[1, 1]),
            "cx": float(self.K[0, 2]),
            "cy": float(self.K[1, 2]),
        }


# ============================================================
# 便捷函数
# ============================================================

def project_points(points_lidar, K, T_lidar2cam, img_w, img_h):
    """
    一行调用版：lidar 点云 → 像素坐标
    """
    proj = LidarToCamera(K, T_lidar2cam, img_w, img_h)
    return proj.project(points_lidar)


def euler_to_rotation(yaw, pitch, roll):
    """欧拉角 (ZYX 顺序) → 旋转矩阵"""
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cr, sr = np.cos(roll), np.sin(roll)

    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    return Rz @ Ry @ Rx


def construct_extrinsic(rotation_euler, translation):
    """欧拉角 + 平移 → 4×4 外参矩阵"""
    R = euler_to_rotation(*rotation_euler)
    t = np.array(translation).reshape(3, 1)
    return np.vstack([
        np.hstack([R, t]),
        np.array([[0, 0, 0, 1]])
    ])


# ============================================================
# 内部工具
# ============================================================

def _pose_from_nuscenes(quat, trans):
    """
    nuScenes 四元数 (w, x, y, z) + 平移 → 4×4 变换矩阵
    """
    w, x, y, z = quat
    R = np.array([
        [1 - 2*(y**2 + z**2), 2*(x*y - z*w),     2*(x*z + y*w)],
        [2*(x*y + z*w),       1 - 2*(x**2 + z**2), 2*(y*z - x*w)],
        [2*(x*z - y*w),       2*(y*z + x*w),     1 - 2*(x**2 + y**2)]
    ])
    T = np.vstack([np.hstack([R, np.array(trans).reshape(3, 1)]),
                   np.array([[0, 0, 0, 1]])])
    return T


# ============================================================
# 可视化
# ============================================================

def visualize(points_lidar, pixels, depth, img_w, img_h,
              save_path="projection_result.png"):
    """
    双栏可视化：左 BEV 俯视 / 右 图像平面投影
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    ax = axes[0]
    ax.scatter(points_lidar[:, 0], points_lidar[:, 1],
              c="#37d6e6", s=1, alpha=0.3)
    ax.set_xlabel("X (前方, m)")
    ax.set_ylabel("Y (左右, m)")
    ax.set_title("BEV 视角 (Lidar)")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.invert_yaxis()

    ax = axes[1]
    ax.imshow(np.zeros((img_h, img_w, 3)), cmap="gray",
              extent=[0, img_w, img_h, 0])
    if len(pixels) > 0:
        sc = ax.scatter(pixels[:, 0], pixels[:, 1],
                       c=depth, s=1, alpha=0.6, cmap="viridis")
        plt.colorbar(sc, ax=ax, label="深度 (m)")
    ax.set_xlim(0, img_w)
    ax.set_ylim(img_h, 0)
    ax.set_xlabel("u (像素)")
    ax.set_ylabel("v (像素)")
    ax.set_title(f"图像平面投影 ({len(pixels)} 个点)")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"✅ 可视化已保存: {save_path}")


# ============================================================
# CLI
# ============================================================

def _axis_transform():
    """
    nuScenes lidar/ego → camera 坐标轴变换

    lidar/ego: x=前, y=左, z=上
    camera:    x=右, y=下, z=前
    => cam_x = -lidar_y, cam_y = -lidar_z, cam_z = lidar_x
    """
    return np.array([
        [ 0, -1,  0],
        [ 0,  0, -1],
        [ 1,  0,  0],
    ])


def _build_demo_calibration(img_w, img_h):
    """构造一组模拟标定参数（含 nuScenes 轴变换）"""
    fx = fy = 1266.4
    cx, cy = img_w / 2, img_h / 2
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

    # 安装位姿的旋转部分（小角度近似无旋转）
    R_install = euler_to_rotation(0.01, -0.02, 0.005)
    # 完整旋转 = 安装旋转 @ 轴变换
    R = R_install @ _axis_transform()
    # 平移：相机相对 lidar（前方1.5m, 上方1.6m）
    t = np.array([1.5, 0.0, 1.6]).reshape(3, 1)
    T = np.vstack([np.hstack([R, t]), np.array([[0, 0, 0, 1]])])

    return K, T


def _build_demo_pointcloud(num_points):
    """构造模拟 lidar 点云"""
    rng = np.random.default_rng(42)
    ground_x = rng.uniform(1, 40, num_points)
    ground_y = rng.uniform(-15, 15, num_points)
    ground_z = np.full(num_points, -1.2)

    obs = np.array([
        [8,  -2, 0.5], [8,  -1, 0.5], [8,  -1, 1.0], [8,  -1, 1.5],
        [15,  3, 0.5], [15,  3, 1.0], [15,  4, 0.5],
        [20,  0, 0.5], [20,  0, 1.0], [20,  1, 1.0], [20, -1, 1.0],
        [25,  2, 0.5], [25, -2, 0.5], [25,  0, 1.5],
    ])

    return np.vstack([
        np.column_stack([ground_x, ground_y, ground_z]),
        obs
    ]).astype(np.float64)


def main():
    parser = argparse.ArgumentParser(
        description="激光雷达点云 → 相机像素坐标投影"
    )
    parser.add_argument("--num-points", type=int, default=1000,
                        help="模拟点云数量")
    parser.add_argument("--img-w", type=int, default=1600,
                        help="图像宽度")
    parser.add_argument("--img-h", type=int, default=900,
                        help="图像高度")
    parser.add_argument("--visualize", action="store_true",
                        help="生成可视化图")
    parser.add_argument("--calibration", type=str, default=None,
                        help="标定参数 JSON 文件（可选，含 K 和 T_lidar2cam）")
    args = parser.parse_args()

    # 构建标定
    if args.calibration and os.path.exists(args.calibration):
        with open(args.calibration) as f:
            cfg = json.load(f)
        K = np.array(cfg["K"])
        T = np.array(cfg["T_lidar2cam"])
        print(f"✅ 已加载标定文件: {args.calibration}")
    else:
        K, T = _build_demo_calibration(args.img_w, args.img_h)
        print("ℹ️  使用模拟标定参数")

    # 构建投影器
    proj = LidarToCamera(K, T, args.img_w, args.img_h)

    print("\n【内参 K】")
    print(K)
    print(f"\n【外参 T_lidar2cam】")
    print(T)

    # 构造点云
    points = _build_demo_pointcloud(args.num_points)
    print(f"\n【输入】{len(points)} 个 lidar 点")

    # 投影
    pixels, depth, mask = proj.project(points, return_depth=True)

    print(f"\n【输出】{len(pixels)} 个点落在图像内 "
          f"({mask.sum()}/{len(points)}, "
          f"{mask.sum()/len(points)*100:.1f}%)")
    if len(pixels) > 0:
        print(f"  像素 u: [{pixels[:, 0].min():.1f}, {pixels[:, 0].max():.1f}]")
        print(f"  像素 v: [{pixels[:, 1].min():.1f}, {pixels[:, 1].max():.1f}]")
        print(f"  深度 Zc: [{depth.min():.1f}, {depth.max():.1f}] m")

    # 验证外参变换维度
    cam = proj.lidar_to_camera(points)
    print(f"\n【维度验证】")
    print(f"  lidar (N,3): {points.shape}")
    print(f"  camera (N,3): {cam.shape}")
    print(f"  pixel (M,2): {pixels.shape}")

    # 可视化
    if args.visualize:
        visualize(points, pixels, depth, args.img_w, args.img_h)

    print("\n✅ 完成")


if __name__ == "__main__":
    main()
