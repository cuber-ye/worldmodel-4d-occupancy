import json, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

with open("docs/loss_curve.json") as f:
    d = json.load(f)

losses = d["losses"]
final = d.get("final_loss", losses[-1])

fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(losses, color="#c0392b", linewidth=0.8, alpha=0.8)
ax.set_xlabel("batch")
ax.set_ylabel("loss")
ax.set_title(f"训练 loss 曲线（最终 {final:.4f}，共 {len(losses)} batches）", fontsize=12)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("docs/loss_curve.png", dpi=150, bbox_inches="tight")
print("saved docs/loss_curve.png")