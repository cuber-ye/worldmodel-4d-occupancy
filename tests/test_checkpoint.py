import sys, os, torch

# 关键：把项目根目录（tests\ 的上层）加入搜索路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.train_mnist import TinyCNN


def test_load():
    ckpt = torch.load("runs/day01/run1/ckpt.pt", map_location="cpu")
    assert "model_state" in ckpt, "checkpoint 缺少 model_state"
    m = TinyCNN()
    m.load_state_dict(ckpt["model_state"])
    m.eval()
    x = torch.randn(4, 1, 28, 28)
    with torch.no_grad():
        out = m(x)
    assert out.shape == (4, 10), f"输出形状异常: {out.shape}"
    print("CHECKPOINT OK  final_loss=", round(ckpt["final_loss"], 4))


if __name__ == "__main__":
    test_load()