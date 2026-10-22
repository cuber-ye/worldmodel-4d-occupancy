import torch

# ============================================================
# A0 热身：BEV 特征张量，shape = (B, H, W, C)
# 含义：8 帧、每帧 32 行 64 列、每格 3 个通道
# ============================================================
x = torch.randn(8, 32, 64, 3)
print("原始 shape:", x.shape)

# A1: 取第 3 帧、第 10~20 行、第 30~40 列
a1 = x[3, 10:20, 30:40, :]
print("A1 shape:", a1.shape, "  # 你的答案写在这里")

# A2: 同上，但保留 batch 维 → 能直接喂给 Conv2d
a2 = x[3:4, 10:20, 30:40, :]
print("A2 shape:", a2.shape, "  # 应该是 (1, 10, 10, 3)")

# A3: 隔行取样，等价于把 BEV 分辨率砍半
a3 = x[:, 0:32:2, 0:64:2, :]
print("A3 shape:", a3.shape, "  # 应该是 (8, 16, 32, 3)")

# A4: 只取第 0 通道 → 常用于可视化某一层特征
a4 = x[3, :, :, 0]
print("A4 shape:", a4.shape, "  # 应该是 (32, 64)")

# A5: 负索引，从末尾取
a5 = x[-1, -5:, -5:, -1]
print("A5 shape:", a5.shape, "  # 应该是 (5, 5)")

# A6: 布尔掩码，取出满足条件的所有格子
mask = x[3, :, :, 0] > 0.5     # shape=(32, 64) 的 True/False 矩阵
a6 = x[3, :, :, :][mask]        # 用 mask 去取
print("mask 中 True 的个数:", mask.sum().item())
print("A6 shape:", a6.shape, "  # 应该是 (N, 3)，N = True 的个数")

# A7: 升维，两种方法
single = x[3]                          # shape=(32, 64, 3)
b1 = single.unsqueeze(0)                # 方法1：在第 0 维插个 1
b2 = single[None, ...]                  # 方法2：用 None 索引
print("unsqueeze:", b1.shape)
print("None 索引:", b2.shape)
print("两者相等:", torch.equal(b1, b2))

# A8: 展平
v1 = x.view(8, -1)
v2 = x.reshape(8, -1)
print("view  :", v1.shape)
print("reshape:", v2.shape)

# 现在先 transpose 再试
xt = x.transpose(1, 2)   # 交换 H 和 W → shape=(8, 64, 32, 3)
print("\ntranspose 后 is_contiguous:", xt.is_contiguous())
try:
    vt = xt.view(8, -1)
    print("view 成功:", vt.shape)
except Exception as e:
    print("view 报错:", type(e).__name__)

rt = xt.reshape(8, -1)
print("reshape 成功:", rt.shape)

# A9 修正：给每一列位置一个权重（按 W 维度加权）
w = torch.randn(64, 1)          # shape=(64, 1)
out = x + w                     # ✅ 成功
print("A9 正确写法:", out.shape, "  # (8, 32, 64, 3)")

# A10: 综合题，模拟 Occupancy 网格操作
occ = torch.randn(4, 200, 200, 16)
result = occ[-1, 50:150, 50:150, 0:3]
print("A10 shape:", result.shape, "  # 应该是 (100, 100, 3)")