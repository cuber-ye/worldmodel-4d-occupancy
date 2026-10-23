# 相机坐标变换公式速查

## 1. 完整变换链路

```
lidar/ego 3D (x, y, z)
        │  第②步 外参 T_lidar2cam = [R|t]
        ▼
camera 3D (Xc, Yc, Zc)
        │  第③步 内参 K + 透视除法
        ▼
pixel 2D (u, v)
```

---

## 2. 第②步：外参变换（3D → 3D）

**作用：** 把点从一个坐标系搬到相机坐标系

### 矩阵形式

$$
\begin{bmatrix} X_c \\ Y_c \\ Z_c \\ 1 \end{bmatrix}
=
\begin{bmatrix} R_{3×3} & t_{3×1} \\ 0 & 1 \end{bmatrix}
\cdot
\begin{bmatrix} x \\ y \\ z \\ 1 \end{bmatrix}
$$

### 展开

$$
\begin{aligned}
X_c &= r_{11}x + r_{12}y + r_{13}z + t_x \\
Y_c &= r_{21}x + r_{22}y + r_{23}z + t_y \\
Z_c &= r_{31}x + r_{32}y + r_{33}z + t_z
\end{aligned}
$$

### 维度追踪

| 步骤 | 输入 | 操作 | 输出 |
|---|---|---|---|
| 齐次化 | $(N, 3)$ | 补 $w=1$ | $(N, 4)$ |
| 矩阵乘 | $(N, 4)$ | @ $T^T$ | $(N, 4)$ |
| 去齐次 | $(N, 4)$ | 取 $[:, :3]$ | $(N, 3)$ |

### 旋转矩阵 $R$ 的性质

| 性质 | 公式 | 用途 |
|---|---|---|
| 正交性 | $R^T R = I$ | 保长度不变 |
| 行列式 | $\det(R) = 1$ | 保右手系 |
| 逆即转置 | $R^{-1} = R^T$ | 反变换不用重算 |

### 欧拉角 → 旋转矩阵（ZYX 顺序）

```python
def euler_to_rotation(yaw, pitch, roll):
    Rz = [[cos(y), -sin(y), 0], [sin(y), cos(y), 0], [0, 0, 1]]
    Ry = [[cos(p), 0, sin(p)], [0, 1, 0], [-sin(p), 0, cos(p)]]
    Rx = [[1, 0, 0], [0, cos(r), -sin(r)], [0, sin(r), cos(r)]]
    return Rz @ Ry @ Rx
```

### nuScenes 轴变换（lidar/ego → camera）

```
cam_x = -ego_y
cam_y = -ego_z
cam_z =  ego_x
```

---

## 3. 第③步：内参投影（3D → 2D）

**作用：** 把相机坐标系下的3D点投影到图像像素平面

### 内参矩阵 $K$

$$
K =
\begin{bmatrix}
f_x & 0   & c_x \\
0   & f_y & c_y \\
0   & 0   & 1
\end{bmatrix}
$$

| 元素 | 含义 |
|---|---|
| $f_x, f_y$ | 等效焦距（像素），通常 $f_x \approx f_y$ |
| $c_x, c_y$ | 光心（主点）像素坐标 |

### 投影 + 透视除法

$$
\begin{bmatrix} u' \\ v' \\ w' \end{bmatrix}
= K \cdot
\begin{bmatrix} X_c \\ Y_c \\ Z_c \end{bmatrix}
=
\begin{bmatrix}
f_x X_c + c_x Z_c \\
f_y Y_c + c_y Z_c \\
Z_c
\end{bmatrix}
$$

$$
u = \frac{u'}{w'} = \frac{f_x X_c}{Z_c} + c_x,
\quad
v = \frac{v'}{w'} = \frac{f_y Y_c}{Z_c} + c_y
$$

### 维度追踪

| 步骤 | 输入 | 操作 | 输出 |
|---|---|---|---|
| 矩阵乘 | $(N, 3)$ | @ $K^T$ | $(N, 3)$ 齐次像素 |
| 取深度 | $(N, 3)$ | $[:, 2]$ | $(N,)$ 深度 |
| 透视除法 | $(N, 3)$ | 前两列 / 深度 | $(N, 2)$ 像素坐标 |

### 为什么要除以 $Z_c$

近大远小。同一方向上的点（比例相同）投影到归一化平面是同一位置，除以深度 $Z_c$ 就是"归一化到单位深度"。

---

## 4. 完整链路（一步到位）

### 公式

$$
\begin{bmatrix} u \\ v \\ 1 \end{bmatrix}
=
K \cdot R \cdot
\begin{bmatrix} x \\ y \\ z \end{bmatrix}
+ K \cdot t
$$

或更一般地：

$$
p_{2d} = K \cdot T_{\text{lidar2cam}} \cdot P_{\text{lidar}}
$$

然后做透视除法。

### 代码

```python
def lidar_to_pixel(points_lidar, K, T_lidar2cam, img_w, img_h):
    # 第②步
    N = len(points_lidar)
    hom = np.hstack([points_lidar, np.ones((N, 1))])  # (N, 4)
    cam = hom @ T_lidar2cam.T                          # (N, 4)
    cam = cam[:, :3]                                   # (N, 3)

    # 第③步
    proj = cam @ K.T                                   # (N, 3)
    Zc = proj[:, 2]
    u = proj[:, 0] / Zc
    v = proj[:, 1] / Zc

    # 过滤
    valid = (Zc > 0) & (u >= 0) & (u < img_w) & (v >= 0) & (v < img_h)
    return np.stack([u, v], axis=1)[valid]
```


## 5. 边界过滤清单

| 条件 | 原因 |
|---|---|
| $Z_c > 0$ | 相机只能看到前方，$Z_c < 0$ 在背后 |
| $0 \leq u < W$ | 超出图像宽度不可见 |
| $0 \leq v < H$ | 超出图像高度不可见 |

---

## 6. 坐标系约定速查

### nuScenes

| 坐标系 | X | Y | Z |
|---|---|---|---|
| lidar / ego | 前 | **左** | 上 |
| camera | **右** | **下** | 前 |

⚠️ lidar 的 Y 朝左、camera 的 X 朝右，两者有轴翻转，不是纯平移关系。

### KITTI

| 坐标系 | X | Y | Z |
|---|---|---|---|
| velodyne / camera | 前 | **左** | **上** |

KITTI 的 camera 是 X右 Y下 Z前 还是 X前 Y左 Z上，取决于用的是 `cam0` 还是 `velodyne` 坐标系，读数据时务必核对标定文件。

---

## 7. 常见坑

1. **轴翻转漏了**：直接拿 nuScenes 的 `calibrated_sensor` 外参乘 lidar 点，忘了 lidar 和 ego 的轴方向不同
2. **取逆搞反**：`calibrated_sensor` 给的是 sensor→ego，要做 ego→camera 需要取逆
3. **忘记透视除法**：只做了 $K \cdot R \cdot P + K \cdot t$，没除以 $Z_c$，投影全错
4. **齐次维度没去掉**：变换后取 $[:, :3]$，不是 $[:, 3]$
5. **深度为负没过滤**：导致投影点镜像到画面错误位置

---

## 8. 一句话总结

> 内参 $K$ 负责"相机怎么看"（焦距 + 光心），外参 $[R|t]$ 负责"相机在哪看"（旋转 + 平移），齐次坐标让平移也能写成矩阵乘法，透视除法让远近有大小之分。
