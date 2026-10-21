import torch, matplotlib
matplotlib.use("Agg")   # Windows 无显示环境必须加，否则卡死
import matplotlib.pyplot as plt
from torchvision import datasets, transforms

ds = datasets.MNIST(
    "data/raw", train=True, download=True,
    transform=transforms.ToTensor()
)

fig, axes = plt.subplots(3, 8, figsize=(12, 5))
for i, ax in enumerate(axes.flat):
    img, label = ds[i * 800]          # 隔 800 张取一张，避免全是同一个数字
    ax.imshow(img.squeeze(), cmap="gray")
    ax.set_title(str(label), fontsize=10)
    ax.axis("off")
plt.tight_layout()
plt.savefig("docs/mnist_samples.png", dpi=120, bbox_inches="tight")
print("saved docs/mnist_samples.png")