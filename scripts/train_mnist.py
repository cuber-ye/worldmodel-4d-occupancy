import os, argparse, random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from device import get_device, device_info


def set_seed(seed: int):
    """固定所有随机源。这是世界模型实验可复现的第一道关。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


class TinyCNN(nn.Module):
    """故意写得极简：一个可运行模型 > 一个完美的模型。"""
    def __init__(self, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 128), nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=str, default="runs/day01")
    args = ap.parse_args()

    set_seed(args.seed)
    device = get_device()
    os.makedirs(args.out, exist_ok=True)

    print("device:", device_info())

    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train_ds = datasets.MNIST(root="data", train=True, download=True, transform=tfm)
    # Windows 下 num_workers 必须 0，pin_memory 在 CPU 训练时无意义还占内存
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=0, pin_memory=False)

    model = TinyCNN().to(device)
    opt = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    final_loss = 0.0
    for epoch in range(args.epochs):
        model.train()
        running = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            opt.step()
            running += loss.item()
        final_loss = running / len(train_loader)
        print(f"epoch {epoch}  loss={final_loss:.4f}")

    # 权重 + 训练参数 + 随机种子，三者一起才有复现意义
    torch.save({
        "model_state": model.state_dict(),
        "optimizer_state": opt.state_dict(),
        "epoch": args.epochs,
        "seed": args.seed,
        "final_loss": final_loss,
    }, os.path.join(args.out, "ckpt.pt"))
    print("SAVED", os.path.join(args.out, "ckpt.pt"), "loss=", round(final_loss, 4))


if __name__ == "__main__":
    main()