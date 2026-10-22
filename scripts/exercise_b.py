import torch
import math

# ============================================================
# B1: 朴素交叉熵（正确但数值不稳定）
# ============================================================
def cross_entropy_naive(logits, target):
    """
    logits: 模型原始输出，shape=(N, C)
    target: 真实标签，shape=(N,)，每个是 0~C-1 的整数
    """
    # 第 1 步：softmax
    exp = torch.exp(logits)              # 每个元素取 e^x
    prob = exp / exp.sum(dim=1, keepdim=True)   # 除以每行总和

    # 第 2 步：取真实类的那个概率
    N = logits.shape[0]
    true_class_prob = prob[torch.arange(N), target]   # 取出"真实类"那一列

    # 第 3 步：负对数
    loss = -torch.log(true_class_prob)
    return loss.mean()


# 测试：3 个样本，10 个类别
torch.manual_seed(42)
logits = torch.randn(3, 10)
target = torch.tensor([3, 0, 7])

loss = cross_entropy_naive(logits, target)
print("B1 朴素实现 loss =", round(loss.item(), 4))

# ============================================================
# B2: 和 nn.CrossEntropyLoss 对比
# ============================================================
criterion = torch.nn.CrossEntropyLoss()
official = criterion(logits, target)
print("官方 loss      =", round(official.item(), 4))
print("差值           =", round(abs(loss.item() - official.item()), 6))

# ============================================================
# B3: 制造数值溢出 —— 亲眼看到 nan
# ============================================================
#logits_big = torch.tensor([[1000.0, 0.0],       # 第 0 类 logit 巨大
#                          [2000.0, 0.0]])       # 更极端
#target_big = torch.tensor([0, 0])
#
#print("\nexp(1000) =", math.exp(1000))   # 这行会怎样？

# ============================================================
# B5: 数值稳定版交叉熵
# ============================================================
def cross_entropy_stable(logits, target):
    # 关键一步：每个 logit 减去这一行的最大值
    # 这样最大的 logit 变成 0，其他的变成负数
    logits_max = logits.max(dim=1, keepdim=True).values
    logits_shift = logits - logits_max          # ← 核心修复

    # 现在最大的 exp 是 e^0 = 1，绝不会溢出
    exp = torch.exp(logits_shift)
    prob = exp / exp.sum(dim=1, keepdim=True)

    N = logits.shape[0]
    true_class_prob = prob[torch.arange(N), target]
    loss = -torch.log(true_class_prob)
    return loss.mean()

# ============================================================
# B6: 验证修复
# ============================================================
print("\n=== B6 数值稳定性验证 ===")

# 极端 logits
logits_extreme = torch.tensor([[1000.0, 0.0, -500.0],
                               [2000.0, 0.0, -1000.0],
                               [50000.0, 0.0, -50000.0]])
target_extreme = torch.tensor([0, 0, 0])

loss_naive = cross_entropy_naive(logits_extreme, target_extreme)
loss_stable = cross_entropy_stable(logits_extreme, target_extreme)
official = torch.nn.CrossEntropyLoss()(logits_extreme, target_extreme)

print("logits 最大值:", logits_extreme.max().item())
print("朴素版 loss   :", loss_naive.item())       # nan
print("稳定版 loss   :", loss_stable.item())       # 一个正常的数
print("官方版 loss   :", official.item())          # 和稳定版一样
print("稳定 vs 官方差值:", abs(loss_stable.item() - official.item()))

# ============================================================
# B7 修正：完整验收
# ============================================================
print("\n=== B7 完整验收 ===")
max_diff = 0.0
for i in range(1000):
    N = torch.randint(2, 64, (1,)).item()
    C = torch.randint(2, 100, (1,)).item()
    scale = torch.randint(1, 10, (1,)).item()   # ← 改成 10
    logits = torch.randn(N, C) * scale
    target = torch.randint(0, C, (N,))

    stable = cross_entropy_stable(logits, target)
    official = torch.nn.CrossEntropyLoss()(logits, target)

    diff = abs(stable.item() - official.item())
    max_diff = max(max_diff, diff)

print(f"1000 次随机测试，最大差值 = {max_diff:.2e}")
print("验收标准: < 1e-6 →", "✅ 通过" if max_diff < 1e-6 else "❌ 失败")