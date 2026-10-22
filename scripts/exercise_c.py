import torch

print("="*60)
print("C1: 手写前向传播")
print("="*60)

# ---- 参数（固定值，方便手算核对）----
W1 = torch.tensor([[1., 2.],
                   [3., 4.],
                   [5., 6.]])      # shape (3, 2)
b1 = torch.tensor([0.1, 0.2, 0.3])  # shape (3,)
W2 = torch.tensor([[1., 2., 3.]])    # shape (1, 3)
b2 = torch.tensor([0.5])             # shape (1,)
x  = torch.tensor([1., 2.])          # shape (2,)

# ---- 前向 ----
z1 = W1 @ x + b1
print("z1 = W1 @ x + b1 =", z1.tolist())
# z1 = [1 * 1+2 * 2+0.1, 3 * 1+4 * 2+0.2, 5 * 1+6 * 2+0.3]
#    = [5.1, 11.2, 17.3]

a1 = torch.relu(z1)
print("a1 = relu(z1)  =", a1.tolist())
# relu: 负数变0，正数不变 → 全正，所以不变

z2 = W2 @ a1 + b2
print("z2 = W2 @ a1 + b2 =", z2.tolist())
# z2 = 1 * 5.1 + 2 * 11.2 + 3 * 17.3 + 0.5 = 81.3

# ---- 损失：假设真实值 y_true=10，用 MSE ----
y_true = torch.tensor([10.])
loss = 0.5 * (z2 - y_true)**2
print("loss = 0.5*(z2 - 10)^2 =", loss.item())
# loss = 0.5 * (81.3 - 10)^2 = 0.5 * 71.3^2 = 2540.845

# ---- dL/dz2：loss 对最后一层输入的导数 ----
dl_dz2 = (z2 - y_true).squeeze()   # 标量
print("dL/dz2 = z2 - y =", dl_dz2.item())

# ---- dL/dW2 和 dL/db2 ----
# z2 = W2 @ a1 + b2，W2 是 (1,3)，a1 是 (3,)
# dL/dW2 = dL/dz2 * a1，结果要 (1,3)
dl_dW2 = dl_dz2 * a1.unsqueeze(0)   # (1,3)
dl_db2 = dl_dz2                     # 标量
print("dL/dW2 =", dl_dW2.tolist())
print("dL/db2 =", dl_db2.item())

# ---- dL/da1：往回传 ----
# z2 = W2 @ a1 + b2 → dL/da1 = W2^T @ dL/dz2
# W2 是 (1,3)，W2.T 是 (3,1)，dL/dz2 是标量
dl_da1 = W2.T.squeeze() * dl_dz2    # shape (3,)
print("dL/da1 =", dl_da1.tolist())

# ---- dL/dz1：ReLU 的导数 ----
dl_dz1 = dl_da1 * (z1 > 0).float()  # shape (3,)
print("dL/dz1 =", dl_dz1.tolist())

# ---- dL/dW1 和 dL/db1 ----
# z1 = W1 @ x + b1，x 是 (2,)
dl_dW1 = torch.outer(dl_dz1, x)     # (3,2) ✅ 现在 dl_dz1 是 1D
dl_db1 = dl_dz1
print("dL/dW1 =", dl_dW1.tolist())
print("dL/db1 =", dl_db1.tolist())

print("\n" + "="*60)
print("C3: torch.autograd 自动求导（作为参考答案）")
print("="*60)

# 重新构建计算图（要用 requires_grad）
W1 = torch.tensor([[1., 2.], [3., 4.], [5., 6.]], requires_grad=True)
b1 = torch.tensor([0.1, 0.2, 0.3], requires_grad=True)
W2 = torch.tensor([[1., 2., 3.]], requires_grad=True)
b2 = torch.tensor([0.5], requires_grad=True)
x  = torch.tensor([1., 2.])

z1 = W1 @ x + b1
a1 = torch.relu(z1)
z2 = W2 @ a1 + b2
y_true = torch.tensor([10.])
loss = 0.5 * (z2 - y_true)**2

loss.backward()   # ← 这一行自动算出所有 .grad

print("torch dW1 =", W1.grad.tolist())
print("torch db1 =", b1.grad.tolist())
print("torch dW2 =", W2.grad.tolist())
print("torch db2 =", b2.grad.tolist())

print("\n" + "="*60)
print("C4: 参数更新（SGD 一步）")
print("="*60)

lr = 0.001

with torch.no_grad():   # 更新时不追踪梯度
    W1_new = W1 - lr * W1.grad
    b1_new = b1 - lr * b1.grad
    W2_new = W2 - lr * W2.grad
    b2_new = b2 - lr * b2.grad

print("W1 更新前:", W1.tolist())
print("W1 更新后:", W1_new.tolist())
print("b2 更新前:", b2.tolist())
print("b2 更新后:", b2_new.tolist())

print("\n" + "="*60)
print("C5: 完整验收 —— 随机测试 1000 次")
print("="*60)

def manual_backward(W1, b1, W2, b2, x, y_true):
    """手写反向传播，返回所有梯度"""
    z1 = W1 @ x + b1
    a1 = torch.relu(z1)
    z2 = W2 @ a1 + b2
    loss = 0.5 * (z2 - y_true)**2

    dl_dz2 = (z2 - y_true).squeeze()          # ← 标量
    dl_dW2 = dl_dz2 * a1.unsqueeze(0)
    dl_db2 = dl_dz2
    dl_da1 = W2.T.squeeze() * dl_dz2          # ← (3,)
    dl_dz1 = dl_da1 * (z1 > 0).float()
    dl_dW1 = torch.outer(dl_dz1, x)           # dl_dz1 必须是 1D
    dl_db1 = dl_dz1

    # 调试断言：任何一个 shape 不对都会立刻报错
    assert dl_dz2.dim() == 0, f"dl_dz2 不是标量: {dl_dz2.shape}"
    assert dl_da1.shape == (3,), f"dl_da1 形状错: {dl_da1.shape}"
    assert dl_dz1.shape == (3,), f"dl_dz1 形状错: {dl_dz1.shape}"
    assert dl_dW1.shape == (3, 2), f"dl_dW1 形状错: {dl_dW1.shape}"

    return {"W1": dl_dW1, "b1": dl_db1, "W2": dl_dW2, "b2": dl_db2}

max_rel_err = 0.0
for i in range(1000):
    torch.manual_seed(i)
    W1 = torch.randn(3, 2, requires_grad=True)
    b1 = torch.randn(3, requires_grad=True)
    W2 = torch.randn(1, 3, requires_grad=True)
    b2 = torch.randn(1, requires_grad=True)
    x  = torch.randn(2)
    y  = torch.randn(1)

    # autograd 版本
    z1 = W1 @ x + b1
    a1 = torch.relu(z1)
    z2 = W2 @ a1 + b2
    loss = 0.5 * (z2 - y)**2
    loss.backward()

    # 手算版本（注意：传的是原始 W1/b1/W2/b2，不用 detach）
    manual = manual_backward(W1, b1, W2, b2, x, y)

    # 对比
    for name in ["W1", "b1", "W2", "b2"]:
        true_grad = {"W1": W1.grad, "b1": b1.grad,
                     "W2": W2.grad, "b2": b2.grad}[name]
        err = torch.abs(manual[name] - true_grad).max().item()
        rel = err / (torch.abs(true_grad).max().item() + 1e-8)
        max_rel_err = max(max_rel_err, rel)

print(f"1000 次测试，最大相对误差 = {max_rel_err:.2e}")
print("验收标准: < 1e-4 →", "✅ 通过" if max_rel_err < 1e-4 else "❌ 失败")