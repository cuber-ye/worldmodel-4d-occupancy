import torch, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]   # 加这行
plt.rcParams["axes.unicode_minus"] = False              # 加这行，让负号正常显示
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from device import get_device
from train_mnist import TinyCNN

device = get_device()
ds = datasets.MNIST("data/raw", train=False, transform=transforms.ToTensor())
dl = DataLoader(ds, batch_size=256, shuffle=False, num_workers=0)

model = TinyCNN().to(device)
ckpt = torch.load("runs/day01/run1/ckpt.pt", map_location=device)
model.load_state_dict(ckpt["model_state"])
model.eval()

cm = np.zeros((10, 10), dtype=int)
with torch.no_grad():
    for x, y in dl:
        p = model(x.to(device)).argmax(1).cpu()
        for t, pred in zip(y, p):
            cm[t.item(), pred.item()] += 1

fig, ax = plt.subplots(figsize=(8, 7))
im = ax.imshow(cm, cmap="YlOrRd")
ax.set_xticks(range(10)); ax.set_yticks(range(10))
ax.set_xlabel("预测"); ax.set_ylabel("真实")
ax.set_title("混淆矩阵  (对角线越亮 = 越准)", fontsize=13)
for i in range(10):
    for j in range(10):
        if cm[i, j] > 0:
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black",
                    fontsize=8)
fig.colorbar(im, ax=ax)
plt.tight_layout()
plt.savefig("docs/confusion_matrix.png", dpi=120, bbox_inches="tight")
print("saved docs/confusion_matrix.png")

# 找出最容易混淆的一对
off = cm.copy()
np.fill_diagonal(off, 0)
i, j = np.unravel_index(off.argmax(), off.shape)
print(f"最容易混淆: 真实 {i} 被判成 {j}，共 {off[i, j]} 次")