# Day 01 (2026-10-22)

## 环境
| 项目 | 本地 Windows | Colab |
|---|---|---|
| OS | Windows 11 25H2 | Ubuntu 容器 |
| Python | 3.10.x | 3.10.x (预装) |
| torch | 2.14.1+cpu | 2.14.1+cu130 |
| 设备 | CPU (AMD RX 6500 XT, 无 CUDA) | Tesla T4 16GB |
| CUDA | False | True |

## 依赖清单
- 锁定: requirements.lock
- 复原: `python -m venv .venv && .\.venv\Scripts\Activate.ps1 && pip install -r requirements.lock`

## 实验结果
| 环境 | final_loss | 3 次一致 |
|---|---|---|
| 本地 CPU | 0.0366 | 是 |
| Colab T4 | 0.0363 | 是 |

两端差异 0.0003 (1e-4 量级)，属 CPU / CUDA 算子实现差异，非种子问题。

## 成功点
- [x] venv 独立隔离，CUDA: False 属预期（非报错）
- [x] MNIST 3 epoch 跑通
- [x] 连续 3 次 loss 完全一致（确定性训练已验证）
- [x] checkpoint 可重新加载并推理
- [x] Colab T4 上同一份脚本跑通
- [x] 两端 torch 版本对齐至 2.14.1

## 卡点（5 条 + 解决方案）
1. venv 建在 C:\Users\62708\ 与项目脱离 → 删掉重建到项目目录
2. .gitignore 必须先于 git add，否则对已跟踪文件失效
3. tests\ 里跑测试报 ModuleNotFoundError → dirname(dirname(__file__))
4. Colab 预装 torch 2.11，与本地 2.14 差 3 个小版本 → 升级对齐
5. test_checkpoint 与 train 默认 out 路径不一致 → 建 config.py 统一

## 双轨说明
本地 = 开发 + 调试 + 小样本验证；Colab = 正式训练 + 多步预测。
切换点: scripts/device.py 的 get_device()，改这一处即可。

## 明天前置
- [ ] 注册 nuScenes 账号（非商业免费）
- [ ] Day02: 张量索引 / 手动 loss / 手动反向传播