import torch, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from device import get_device
from train_mnist import TinyCNN

device = get_device()
ds = datasets.MNIST("data/raw", train=False, transform=transforms.ToTensor())
img, label = ds[0]
x = img.unsqueeze(0).to(device)

model = TinyCNN().to(device)
ckpt = torch.load("runs/day01/run1/ckpt.pt", map_location=device)
model.load_state_dict(ckpt["model_state"])
model.eval()

# 钩子：抓取第一层卷积的输出
feat = {}
def hook(_, __, out):
    feat["out"] = out.detach()

handle = model.net[0].register_forward_hook(hook)
with torch.no_grad():
    _ = model(x)
handle.remove()

f = feat["out"][0].cpu()          # shape: (16, 28, 28)，16 个特征图
fig, axes = plt.subplots(4, 4, figsize=(10, 10))
for i, ax in enumerate(axes.flat):
    ax.imshow(f[i], cmap="hot")
    ax.set_title(f"channel {i}", fontsize=8)
    ax.axis("off")
plt.suptitle(f"第一层卷积的 16 个特征图  (输入数字 = {label})", fontsize=13)
plt.tight_layout()
plt.savefig("docs/features_layer1.png", dpi=120, bbox_inches="tight")
print("saved docs/features_layer1.png")