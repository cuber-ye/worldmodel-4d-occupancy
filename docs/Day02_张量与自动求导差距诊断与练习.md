# Day 02：用三道最小题拆掉“会跑 MNIST”的幻觉

## 摘要

Day 02 的目标不是再训练一个模型，而是做一次**能力体检**：你已经能把 TinyCNN 跑通、得到 `final_loss=0.0366`，说明能照着训练骨架完成一次实验；但从“照着做”到“改得动”，中间缺的是四类可迁移零件——张量索引与形状推理、损失函数的数值语义、计算图与反向传播、参数更新循环。本日采用“诊断 20 分钟 + 三个番茄钟练习 135 分钟 + 回填 25 分钟”的结构；三题分别为张量裁剪与视图、手写稳定交叉熵、两层 MLP 手算梯度。真正的难点是练习 B：先故意让朴素 `exp(logit)` 在 `logit=1000` 时溢出，亲眼看到 `inf/nan`，再由学习者自己推出“分子分母同除 `exp(max)` 等价于各 logit 减去 max”的修复；这比直接看答案更能建立对数值稳定性的直觉。验收不以“代码跑通”为准，而以 `assert`、形状核对、与 `torch.autograd` 对比为准。完成标准应写成：不查资料，能口述并写出张量索引、损失计算、反向传播、参数更新四段流程，其中手写交叉熵与官方 API 的误差小于 `1e-6`，手算梯度与自动求导相对误差小于 `1e-4`。若三个番茄钟超时，保底顺序为：完成 A 基础切片 → 完成 B“nan 实验 + 修复” → 保留 C 的前向与权重梯度；放弃 C 的偏置推导和进阶诊断。

## 你的真实水平不是“K12”，而是“完成了一次正确复现”

**能跑通 TinyCNN，已经证明你会设置环境、读取数据、调用训练接口；但这仍然不等于理解了代码。** 对学习路径而言，这是非常重要的一层分层：代码运行成功只说明当前输入的语法、形状、类型和运行顺序凑得起来；它不说明你能预测下一行张量的形状，也不说明你知道 `loss.backward()` 从哪个标量开始反向、为何要 `optimizer.zero_grad()`。Day 02 要测的正是这层“预测与修改能力”。

建议你把本日看成一次**基线测试**，而不是一章新课。若以下三件事全部成立，才说明复现能力已经转化为修改能力：

1. 看到一个张量形状，能在动笔前写出索引、切片、转置、合并之后的结果形状；
2. 看到一个损失函数，能说明输入是什么、目标是什么、输出是否可微、为何要这样设计；
3. 看到一个报错，能先读形状和 dtype，而不是连续改代码、反复重新运行。

PyTorch 的 `backward()` 依赖动态计算图：把 `requires_grad=True` 的张量交给运算后，结果会保留 `grad_fn`，最后从标量损失反向计算梯度；但张量默认不追踪梯度，梯度也会在同一个 `.grad` 上累加，因此训练循环必须 `optimizer.zero_grad()`，否则新旧梯度会混在一起[1]。你之前“跑过 TinyCNN”的成功，可能恰好落在最常见但最危险的状态：调用顺序是对的，但顺序背后的因果没有说出来。今天就要把这种隐性顺序变成可表述的规则。

## 差距清单：四列自测，逐项打分而不是自我感觉良好

**这份清单的用途是暴露具体缺口。** 每项只评 `0/1/2`：0 表示不会或说错；1 表示照着例子能做，但不能解释；2 表示换数据、换形状、换 API 后仍能正确使用。最终不要只报总分；如果某项只是“复制时能跑”，必须评 1 而不是 2。

### Python：不是“学过语法”，而是能预测名字和对象怎么变

| 自评项 | K12 常见误区 | 正确认知 | 项目落点 |
|---|---|---|---|
| `=`、可变对象、函数参数 | 认为传进函数的列表被“整体保护”，函数里改了外面不变 | `=` 是把名字绑到对象；列表、字典是可变对象，`.append` 等原地操作会影响外部；`x = x + [...]` 则换新对象 | 第5周 ablation 里改 config dict、冻结参数、复制 checkpoint 时，避免误改共享配置 |
| 函数返回与副作用 | 函数里改了参数，忘记 `return`，认为外面自动更新 | 优先用返回值，不用可变全局状态；函数名写清是“新建”还是“原地修改” | 损失构造器、指标函数应返回新张量；训练日志不要靠全局变量累积 |
| 循环与断言行 | 靠 `print` 看结果，没有“机器判对错”的意识 | 用 `assert shape == ...`、`assert torch.allclose(...)` 构造可重复验收 | 第4周 3D 张量调试时间会成倍下降 |
| 随机种子 | 以为 `torch.manual_seed` 能保证 NumPy、Python、CUDA 全部确定 | 通常还要固定 `random`、`numpy.random`，Colab 与本地环境、算子版本也会影响结果 | 第6周云端复现第4周实验，否则 ablation 差异无法归因 |
| 错误阅读顺序 | 看到一大片红字就改第一行，或反复搜索报错原文 | 先看最后一条异常的**最上面、最接近自己文件的那行**；再检查 shape/dtype/requires_grad | 云端大规模训练出现 OOM、shape mismatch 时，能自己定位而非等答案 |
| 文件与依赖 | 把文件命名为 `torch.py`、到处 `pip install`，分不清系统解释器和 `.venv` | 一个项目一个虚拟环境；用 `python -m` 运行；依赖进入 `requirements.lock` | 第6周 Colab/云端镜像迁移，先锁 Python 与 PyTorch 版本 |

Python 中的“赋值即定义”、局部作用域遮蔽外部同名变量，是函数化代码最常见的隐性错误来源；北京大学的计算课程资料也把“哪些同名名字其实不是同一变量”列为基本判断[6]。这一条对你尤其重要：未来项目里 `loss_fn`、`model`、`config` 会被传给不同函数，如果你只记得“变量名”，不记得“它当前绑的是哪个对象、是否可变”，错误会在函数边界悄然发生。

### 张量：不是“能创建张量”，而是能读写形状语言

| 自评项 | K12 常见误区 | 正确认知 | 项目落点 |
|---|---|---|---|
| `shape`、`dim`、`ndim`、`dtype`、`device` | 看到 `(B,T,H,W,C)` 只背“这是四维”，不知道哪一维是时间 | 从左到右依次命名各维；索引前先写出原始 `shape`，索引后逐项推算 | BEV 特征 `(8,32,64,3)` 的第0维通常是 batch/帧，第1、2维是空间，第3维是通道 |
| 切片边界 | 认为 `10:20` 含 20，或把左闭右开写成“从10到20共11个” | `start:stop` 含 `start`，不含 `stop`；`10:20` 的索引为 `10,11,...,19`，共 10 个 | 多尺度特征裁剪、GT 窗口、时序滑窗不能错位 |
| 0 起始与负数索引 | 把“第3帧”当成索引 3；`[-1]` 只记得倒数 | “第3帧”通常索引 2；`-1` 是最后一维元素，`...` 用于省略中间维度 | 取最近历史帧、通道最后一维特征 |
| bool 索引 | 认为 mask 切片后仍保持原二维网格 | `mask` 是同形布尔张量，结果是一维、只保留 True 位置 | 未来做有效性 mask、可行驶区域、可见 voxel 过滤 |
| 高级索引与 `gather`/`index_select` | 把 fancy index 和 view 混为一谈，误以为索引不会复制 | 连续整数步长的基本切片常共享存储；含 bool、整数列表、重复索引的高级索引会产生新张量 | 按 voxel 索引取特征、时序采样、类别重排 |
| `view` 与 `reshape` | 认为两者只是名字不同，可以随便换 | `view` 通常要求存储连续且步长兼容，不复制；`reshape` 更宽松，必要时会复制；改维度前先 `is_contiguous()` | BEV→3D voxel、卷积前后 `permute`/`flatten` 的形状链 |
| `squeeze`/`unsqueeze`/`permute` | 用 `view(-1)` 代替一切维度操作 | `unsqueeze(0)` 增加 batch 维；`squeeze` 移除尺寸为1的维；转置须配合 contiguous 再 view | 单样本推理、通道前置、BEV 高度维转换 |

张量的索引包含基本索引、切片、ellipsis、newaxis，以及整数/布尔高级索引；高级索引与视图在是否共享存储、是否连续方面规则不同。官方文档将 `view` 描述为返回有相同数据但形状不同的张量，强调新的尺寸必须与原有 size/stride 兼容；`reshape` 则会“在可能时返回视图，必要时返回副本”[7]。这里不必背所有边界，但必须掌握一条工程规则：**形状变了以后立刻写 `assert`，别相信记忆。**

### 自动求导：不是“调用 `backward()`”，而是能解释计算图怎么走

| 自评项 | K12 常见误区 | 正确认知 | 项目落点 |
|---|---|---|---|
| `requires_grad` | 认为所有张量自动求导，或只要 `.grad` 存在就能用 | 只有 `requires_grad=True` 的叶子张量、以及由其参与的可追踪运算结果才构建图；常数张量不会自动追踪 | 第4周控制输入、target、mask 是否参与图 |
| `grad_fn` | 只把 `grad_fn` 当显示内容，不知道它代表“我可被反推” | `grad_fn` 记录生成该张量的操作；用户直接创建的叶子张量通常没有它 | 检查中间变量是否仍在图中，定位无梯度问题 |
| 标量反向传播 | 对多维 loss 直接 `.backward()`，不理解 `gradient=` | `loss` 应是标量；`backward()` 从它开始；非标量需传形状匹配的梯度权重 | 多任务 loss、逐元素 loss 要先 reduction |
| 链式法则 | 认为链式法则就是“连续写几个 `.backward()`” | 从最终标量逐层乘局部雅可比；PyTorch 内部计算的是雅可比—向量积 | 手推 MLP、时序损失、自定义 flow loss |
| 梯度清零与累加 | 以为 `backward()` 每次自动覆盖梯度 | 默认梯度会累加到 `.grad`；训练前要 `zero_grad()`；梯度累积是刻意不清除 | 第6周用梯度累积模拟大 batch，但要清楚少调用了多少次 `step()` |
| 计算图释放 | 多次 `backward()` 时不知道 `retain_graph` | 默认释放中间缓存以省内存；再次反向需要保留图，否则报错 | 调试和自定义多路反向；但常态训练不应滥用 |
| 停止追踪 | 以为 `with torch.no_grad()` 只是提速装饰 | 它明确关闭梯度追踪，用于推理、参数更新、评估；不能在有状态训练图中随意包裹前向 | 推理脚本、移动平均、后处理、指标计算 |
| `detach` | 用 `detach()` 逃避报错，却不知道切断了梯度 | 它返回一个不追踪历史的新张量；会把梯度流截断 | teacher forcing、stop-gradient、伪标签设计 |

官方 autograd 资料将计算图描述为张量与 `Function` 互相连接形成的非循环图；反向传播是对雅可比矩阵与上游梯度向量做乘积，而不是只调用一个“魔法函数”[1]。你能跑通 MNIST，说明至少默认路径有效；今天要测的是：**如果你把 `backward` 删掉、把优化器换成手算更新，还能否让同一个模型下降？**

### 基础神经网络：不是“知道 epoch/loss/反向传播”，而是能画出一个最小循环

| 自评项 | K12 常见误区 | 正确认知 | 项目落点 |
|---|---|---|---|
| 前向传播 | 认为模型只是“输入进去、输出出来” | 一层层算 `z=Wx+b`、`a=activation(z)`；每层 shape 都要连续推导 | 第4周 BEV→3D 主干、时序融合、预测头 |
| loss 的对象 | 把 loss 当“分数”，谁小谁好，但不检查目标格式 | 分类 loss 的 logits 通常不预先 softmax；target 类别必须是 `long`，类别维在正确位置 | occupancy 类别、空/占/自由/语义、flow vector 的多个 head |
| 负对数似然 | 只记住“取负号再平均” | NLL 是从模型对真实类别的预测概率取负对数；softmax 给出分布，log 防止数值问题 | 未来多类 occupancy、难例类别、ignore index |
| 梯度下降 | 认为 `lr` 是一个可以随便调的神秘参数 | `θ ← θ − lr · ∇θL`；学习率决定沿负梯度方向走多远 | 云端训练、3D 卷积内存受限时的 batch/梯度累积配置 |
| 优化器 | 认为 `optimizer.step()` 就是 backward | `zero_grad → forward → loss → backward → step → zero_grad` 是一整条链；`step()` 才更新参数 | ablation 不能只换模型，还要冻结随机性与优化器状态 |
| 训练/验证模式 | 把 `model.eval()` 理解为“不计算 loss” | 它切换 dropout、batch norm 等层的行为；仍需 `torch.no_grad()` 节省计算 | 第4、5周公平比较不同结构 |
| 指标与 loss 混用 | 用 accuracy 当 loss、把 loss 当 IoU | loss 必须可微且指导优化；accuracy/IoU 用于人读评估 | 4D Occupancy 必须同时看 occupancy loss、flow 误差、IoU、内存、时延 |
| 过拟合基线 | 认为 loss 下降就是“学到了” | 先让小数据、小模型过拟合，再扩大规模；这是验证实现正确性的第一关 | 第6周云端大模型迁移后确认代码而非数据泄漏 |

## 三个番茄钟怎么切：先把诊断做小，再把三题做成闭环

**三小时不是“看三小时资料”，而是两个半小时写代码和反思。** 建议的时钟切分如下。若使用番茄钟，可把 25 分钟换成 30 分钟，但总休息时间不要超过 30 分钟。

| 时段 | 时长 | 产物 | 验收 |
|---|---:|---|---|
| 诊断 | 20 分钟 | 填完四列差距清单；不看资料做白板预测 | 至少写出三处自己说错的概念 |
| 番茄钟1 | 45 分钟 | `day02/exercise_a_indexing.py` | A 基础断言全绿，能口述各维变化 |
| 休息 | 10 分钟 | 离开屏幕 | 不看答案 |
| 番茄钟2 | 50 分钟 | `day02/exercise_b_cross_entropy.py` | 先制造 `inf/nan`，再修到 `diff<1e-6` |
| 休息 | 10 分钟 | 离开屏幕 | 不看答案 |
| 番茄钟3 | 45 分钟 | `day02/exercise_c_backprop.py` | 参数梯度相对误差 `<1e-4` |
| 回填 | 25 分钟 | `docs/day02.md`、commit、错题卡 | 能在两分钟内口述四个核心流程 |

**如果超时，按“形状 → 稳定性 → 梯度”降级。** 不追求三题都做完，因为匆忙复制答案会把今天变成又一次复制粘贴。

1. 只完成 A 基础切片，并留下 bool/gather/view 的 3 张手绘图；
2. 只完成 B 的“nan 复现 + 减 max 修复 + 官方对比”；
3. 只完成 C 的前向、`dW2`、学习率更新；放弃 `db1/dW1`、多次 backward 和 `gradcheck`；
4. 放弃 `docs/day02.md` 长总结，只写“我卡在哪、我如何修好、明天重做哪一步”。

这样的降级仍然比“三题都复制一遍、明天全部忘掉”更符合你的目标。

## 练习 A：从 BEV 特征裁剪窗口，并分清切片、bool 索引和视图

**张量索引的本质是“先解释每一维是什么，再预测结果形状”。** 本题用 4 维 BEV 特征模拟多帧/多 batch、空间高度、空间宽度和通道，重点不是写对一行索引，而是能解释为何形状变成那样。

### 题目原文

创建以下最小数据：

```python
import torch

torch.manual_seed(0)

# 约定：第0维=帧/样本，第1维=高度(H)，第2维=宽度(W)，第3维=通道(C)
bev = torch.randn(8, 32, 64, 3)
```

完成四个任务：

1. 取**第3帧、第10~20行、第30~40列**的所有通道；先口述结果形状，再写代码并断言正确。
2. 用布尔索引取出 `bev` 中所有大于 `0.3` 的值；解释为什么结果不是二维网格。
3. 用 `torch.index_select` 在第0维取索引 `[1, 2, 3]`；说明结果与 `bev[[1, 2, 3]]` 的语义关系。
4. 使用 `torch.gather` 在第3维取出索引 `0` 和 `2`；解释它为何不是按任意位置自由取。
5. 分别用 `view` 和 `reshape` 把 `bev` 变为 `8 × 6144`；比较是否能工作、是否共享数据，以及 `permute` 后再 `view` 会发生什么。

### 基础答案与形状推演

“第3帧”不是索引 3，而是索引 2；Python 从 0 计数。切片 `10:20` 包括 `10,11,...,19` 共 10 个位置；宽度和通道同理。所以结果为：

```python
window = bev[2, 10:20, 30:40, :]
# 或等价地：
window = bev[2, 10:20, 30:40]
assert window.shape == (10, 10, 3)
```

可以这样解释：第0维只保留一个位置，所以结果不再有“帧”这一维；第1维留下 10 个位置，第2维留下 10 个位置，通道维未切片，保持 3。注意 `bev[2, 10:20, 30:40, :]` 与 `bev[2, 10:20, 30:40]` 在这里结果等价，但前者更清楚地表明“保留所有通道”，是可读性更好的写法。

`bool` 索引题的基础答案：

```python
mask = bev > 0.3
selected = bev[mask]
assert selected.dim() == 1
```

这里不能写成 `assert selected.shape == (?,)`，因为满足条件的元素个数由数据决定，不是由原网格形状预先确定。`mask` 与原 `bev` 同形；PyTorch 会依次把为 `True` 的元素收集成一维张量。它适合做“条件筛选”，不适合做“保留二维空间位置的裁剪”。

`index_select` 题：

```python
indices = torch.tensor([1, 2, 3])
picked = torch.index_select(bev, dim=0, index=indices)
assert picked.shape == (3, 32, 64, 3)
```

它按第0维取不连续索引，结果与 `bev[[1, 2, 3]]` 的语义一致；二者都从帧维选择 3 个“页面”。区别在于：`torch.index_select` 明确写出维度，便于阅读，也能嵌入模型逻辑；方括号高级索引更简洁。对于相邻、连续、步长一致的区间，普通切片通常更合适。

`gather` 题要先构造与输出索引形状相同的索引张量：

```python
channel_idx = torch.tensor([0, 2]).expand(8, 32, 64, 2).clone()
out = torch.gather(bev, dim=3, index=channel_idx)
assert out.shape == (8, 32, 64, 2)
```

`gather` 不是“输入任意坐标、输出任意值”的通用查表器。它的索引张量要和输出形状一致；除 `dim` 之外，其余维度提供位置，指定那一维选择哪一个槽位。因此它是“沿固定维度按索引收集”，不是 `masked_select`，也不是 NumPy 那种任意多整数数组组合。

`view`/`reshape` 题：

```python
v1 = bev.view(8, -1)
v2 = torch.reshape(bev, (8, 32 * 64 * 3))
assert v1.shape == (8, 6144)
assert v2.shape == (8, 6144)
assert torch.equal(v1, v2)
```

`-1` 表示“让 PyTorch 根据其余维度和总元素数自动推断”，而不是“无限大”。本例中 `32×64×3=6144`。`view` 强调不复制视图，`reshape` 更宽松；当张量不连续时，`reshape` 可能复制。若先做 `bev.permute(0,3,1,2)` 再 `.contiguous().view(...)`，元素排列会随维度顺序变化，不能与原 `view` 结果逐元素相等。

### 常见错误与调试方法

| 报错/现象 | 根因 | 处理 |
|---|---|---|
| `IndexError: index 3 out of range` | 把“第3帧”写成索引 3 | 第 k 个位置是索引 `k-1` |
| 结果多了一维 `(1,10,10,3)` | 写成了 `bev[2:3, 10:20, 30:40]` | 整数索引消维，切片保留维 |
| `window.shape=(10,10)` | 只给了三对索引 | 通道维要显式保留，通常写 `:` |
| `selected.shape` 无法预测 | 误以为 bool 索引保持原形状 | 先 `mask.shape`，再理解结果为一维 |
| `RuntimeError: shape mismatch` | `gather` 索引与目标维长度不一致 | 用 `expand` 补齐，并检查 `dim` |
| `view size is not compatible` | 步长或内存布局不兼容 | 先 `permute` 再 `contiguous()`，或改用 `reshape` |
| 结果与原张量不相等 | 混淆 view 与复制 | `torch.equal` 检查；用 `storage()` 判断是否共享存储 |
| 看不出错在哪 | 只 `print(tensor)` | 每步都 `print(x.shape, x.dtype)`；把小例子手算 |

Python 索引本身从 0 开始，且切片左闭右开，这是基础但高频的偏差来源[5]。在 4D Occupancy 里，若约定是 `(B,T,H,W,C)`，却把第0维理解成高度，后面的 `10:20` 行裁错空间范围、损失取错时间维几乎是必然的。

### 验收断言

```python
def test_exercise_a():
    torch.manual_seed(0)
    bev = torch.randn(8, 32, 64, 3)

    window = bev[2, 10:20, 30:40]
    assert window.shape == (10, 10, 3)
    assert torch.equal(window, bev[2, 10:20, 30:40, :])
    # 索引无复制时，数值必须逐元素相同
    assert torch.equal(bev[2, 10, 30, 0:1], window[0, 0, 0:1])

    mask = bev > 0.3
    assert mask.shape == bev.shape
    assert mask.dtype == torch.bool
    assert torch.equal(torch.masked_select(bev, mask), bev[mask])
    assert bev[mask].dim() == 1

    idx = torch.tensor([1, 2, 3])
    picked = torch.index_select(bev, dim=0, index=idx)
    assert picked.shape == (3, 32, 64, 3)
    assert torch.equal(picked, bev[idx])

    cidx = torch.tensor([0, 2]).expand(8, 32, 64, 2).clone()
    gathered = torch.gather(bev, dim=3, index=cidx)
    assert gathered.shape == (8, 32, 64, 2)

    assert bev.view(8, -1).shape == (8, 6144)
    assert torch.reshape(bev, (8, 32 * 64 * 3)).shape == (8, 6144)

    perm = bev.permute(0, 3, 1, 2)
    assert perm.shape == (8, 3, 32, 64)
    assert perm.contiguous().view(8, -1).shape == (8, 6144)
```

### 与 4D Occupancy 的关联

未来第4周，3D 卷积输入可能是 `(B,C,D,H,W)`，输出仍是 `(B,C_out,D_out,H_out,W_out)`；时序融合会在时间维拼接、采样或做注意力。如果今天不练“第几维是什么、切片保留还是删除维度、转置后能否 view”，到时候每出现一个 `RuntimeError: shape mismatch` 都要问人。更危险的是没有报错、只有形状相同：例如本想取第3个历史帧，却取成索引 3；网络仍会训练，但学的是错数据。建议把本项目的张量命名约定固定为 `B,T,H,W,C` 或 `B,C,T,H,W`，写进 `docs/conventions.md`，每次新建张量都注释维度含义。

## 练习 B：先亲手算出 nan，再修复稳定交叉熵

**数值稳定性的关键不是背诵公式，而是理解为什么中间值会先爆炸。** 交叉熵衡量真实分布与预测分布的差异；分类任务常先算 logits，再用 softmax 变成概率，最后取真实类别的负对数。最朴素的写法是直接 `exp(logit)`，但当某些 logit 很大时，`exp` 会超过浮点表示范围。

### 题目原文

编写 `my_cross_entropy(logits, target)`，要求：

- 输入 `logits: (N, C)`，不要预先 softmax；
- 输入 `target: (N,)`，类别索引为 `torch.long`；
- 用稳定 `log_softmax` 加 NLL 实现；
- 默认 `reduction="mean"`；
- 对 `(N, C)` 任意形状都能正确运行；
- 与 `torch.nn.functional.cross_entropy` 的最大绝对差小于 `1e-6`；
- 同时包含“朴素实现”脚本，能证明它会在极端 logit 下产生 `inf`/`nan`。

之后完成三件事：

1. 写一段只有 1 个样本、2 类的输入，手算 `log_softmax`、真实类 log 概率和最终 loss；
2. 构造 `logits=[[1000, 0]]`、目标为类别 0，运行朴素实现，观察 `inf/nan`；
3. 在不查 API 的情况下，推导出“每类减去该样本最大值”的数学依据，并把实现改成稳定版。

### 逐步推演

对一个样本，`softmax` 的公式为：

\[
p_j=\frac{\exp(z_j)}{\sum_k \exp(z_k)}
\]

目标是类别 `y` 时：

\[
L=-\log p_y
  =-\left(z_y-\log\sum_k\exp(z_k)\right)
  =\log\sum_k\exp(z_k)-z_y
\]

问题出在 `exp(1000)`：`float32` 可表示的范围远低于 `e^{1000}`，因此分子或归一化分母会溢出成 `inf`。一旦中间值是 `inf`，后续除法、对数和减法可能得到 `nan`。

修复要利用恒等变换：对所有类同时减去同一个常数 `c`，概率不变，因为分子分母会约掉 `exp(c)`：

\[
\frac{\exp(z_j)}{\sum_k\exp(z_k)}
=
\frac{\exp(z_j-c)}{\sum_k\exp(z_k-c)}
\]

取对数后：

\[
\log p_j
=
(z_j-c)-\log\sum_k\exp(z_k-c)
\]

最优选择通常是 `c = max_k z_k`：这样最大指数项为 `exp(0)=1`，其余指数项 `exp(z_j-max) ≤ 1`，不会因 `exp` 太大溢出；同时 `log(sum)` 至少不会小于 0。这不是“把大数删掉”，而是不改变数学结果的前提下改写计算顺序。

可以用极小例子验证。设 `logits=[[1,2]]`，`target=[1]`：

```python
z = torch.tensor([[1.0, 2.0]])
z = z - z.max(dim=1, keepdim=True).values   # [[0.0, 1.0]]
exp = torch.exp(z)                          # [1.0, 2.718...]
s = exp.sum(dim=1, keepdim=True)
logp = z - torch.log(s)
```

`exp(0)=1`、`exp(1)=e`，和为 `1+e`，`log(1+e)≈1.3133`，真实类别在索引1，因此 `loss=-logp[0,1]≈0.3133`。如果你手算得到同一结果，说明不是只记住了函数名。

### 参考实现

```python
import torch
import torch.nn.functional as F

torch.manual_seed(0)


def stable_log_softmax(logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
    # 沿类别维减去最大值；keepdim 保证广播后维数一致
    z = logits - logits.max(dim=dim, keepdim=True).values
    e = torch.exp(z)
    return z - torch.log(e.sum(dim=dim, keepdim=True))


def my_cross_entropy(
    logits: torch.Tensor,
    target: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    if logits.dim() < 1:
        raise ValueError(f"logits must have at least 1 dim, got {logits.dim()}")
    if target.dtype not in (torch.int64, torch.long):
        raise TypeError("target must contain class indices of integer type")
    if logits.shape[0] != target.shape[0]:
        raise ValueError("batch size of logits and target must match")
    if reduction not in {"mean", "sum", "none"}:
        raise ValueError("reduction must be 'mean', 'sum', or 'none'")

    logp = stable_log_softmax(logits, dim=-1)
    nll = -logp.gather(dim=-1, index=target.unsqueeze(-1)).squeeze(-1)

    if reduction == "mean":
        return nll.mean()
    if reduction == "sum":
        return nll.sum()
    return nll


def naive_cross_entropy(logits, target):
    # 故意不稳定：用于演示 overflow
    probs = torch.exp(logits) / torch.exp(logits).sum(dim=1, keepdim=True)
    return -torch.log(
        probs.gather(dim=1, index=target.unsqueeze(1)).squeeze(1)
    ).mean()


def demo():
    logits = torch.randn(4, 7)
    target = torch.randint(0, 7, (4,))

    mine = my_cross_entropy(logits, target)
    ref = F.cross_entropy(logits, target)
    assert torch.isclose(mine, ref, atol=1e-6, rtol=1e-6)

    big = torch.tensor([[1000.0, 0.0]])
    bad = naive_cross_entropy(big, torch.tensor([0]))
    assert torch.isinf(bad) or torch.isnan(bad)

    good = my_cross_entropy(big, torch.tensor([0]))
    assert torch.isfinite(good)

    print("Day 02 B passed:", mine.item(), ref.item(), good.item())


if __name__ == "__main__":
    demo()
```

在 NumPy 同参数数值复算中，朴素实现遇到 `[[1000,0]]` 会产生 `inf`，稳定实现保持有限；而普通随机浮点样本不必然溢出，所以“看到 nan”必须由极端 logits 触发，不能靠运气[4]。这也是练习的核心：不是相信老师说的“大数会炸”，而是亲自制造一个最小反例。

### 常见错误与调试方法

| 错误/现象 | 根因 | 处理 |
|---|---|---|
| `RuntimeError: expected scalar type` | target 是 `float32` | 目标类别索引用 `torch.long` |
| `target.shape` 比 logits 多一维 | 误把 one-hot 当类别索引 | PyTorch `CrossEntropyLoss` 默认收类别索引，不是 one-hot |
| `loss=nan` | `log(0)`、数值下溢，或直接 `exp` 溢出 | 先做 `z - z.max`；用 `torch.isfinite` 断点 |
| 结果恒为 0 | 把 `softmax(logits)` 传给 CrossEntropy | 官方交叉熵收 logits；你手写的 nll 收 log-probabilities |
| `gather` 索引维不匹配 | `index` 与输出维长度不一致 | 先 `target.unsqueeze(-1)`，再 `gather(dim=-1)` |
| 单个样本 `inf`、mean 仍是 finite | 有限值掩盖了坏样本 | 使用 `reduction="none"` 逐样本检查 |
| 只差 `1e-5` | 使用 `float32` 的舍入边界 | 用 `float64` 做严格对照，并放宽到 `1e-5` |
| 认为减 max 会改变类别 | 只看见“数字变小” | 手写分数上下同除 `exp(max)`，证明概率不变 |

### 验收标准

```python
def test_exercise_b():
    torch.manual_seed(0)
    logits = torch.randn(6, 5, dtype=torch.float64)
    target = torch.randint(0, 5, (6,))

    manual = my_cross_entropy(logits, target)
    ref = F.cross_entropy(logits, target)
    assert torch.isfinite(manual)
    assert torch.abs(manual - ref).item() < 1e-6

    none_out = my_cross_entropy(logits, target, reduction="none")
    assert none_out.shape == (6,)
    assert torch.abs(none_out.mean() - ref).item() < 1e-6

    one = torch.tensor([[1.0, 2.0]])
    t = torch.tensor([1])
    assert torch.isclose(
        my_cross_entropy(one, t),
        -torch.log(torch.softmax(one, dim=1)[0, 1]),
        atol=1e-7,
    )

    bad = torch.tensor([[1000.0, 0.0]])
    assert torch.isfinite(my_cross_entropy(bad, torch.tensor([0])))
    assert torch.isinf(naive_cross_entropy(bad, torch.tensor([0]))) or torch.isnan(
        naive_cross_entropy(bad, torch.tensor([0]))
    )
```

### 与 4D Occupancy 的关联

第4周不仅可能有逐 voxel 的类别损失，还可能有 flow vector、可见性 mask 和时序一致性损失。每个损失头的数值范围不同；如果直接把多个 `exp`、距离或概率拼在一起，一个头的数值尺度会主导优化。数值稳定交叉熵的练习会自然迁移到以下能力：

- 用 `log-sum-exp` 而不是 `log(exp(...).sum())`；
- 用 `reduction="none"` 找到坏样本，而不是让 batch mean 掩盖 nan；
- 对 loss 各分量单独打印、用 mask 平均、避免对无效 voxel 做除法；
- 未来做消融时，不只比较一个总 loss，而是同时比较 occupancy CE、flow MSE、有效样本数。

## 练习 C：手算两层 MLP，让“链式法则”变成一张维度表

**手算梯度不是复古练习，而是让你拥有独立定位错误的能力。** 两层 MLP 已经足以覆盖矩阵乘法、ReLU、softmax、NLL、广播和参数更新；若这里能从公式走到代码，未来理解卷积梯度、时序梯度只是增加维度和算子复杂度。

### 题目原文

实现一个二层分类 MLP：

\[
h=\mathrm{ReLU}(XW_1+b_1)
\]

\[
s=H W_2+b_2
\]

\[
L=\mathrm{CrossEntropy}(s,y)
\]

要求：

- `X: (N, D)`、`W1: (D, H)`、`b1: (H,)`、`W2: (H, C)`、`b2: (C,)`；
- 不使用 `loss.backward()` 计算参数梯度，而是手写 `dW1, db1, dW2, db2`；
- 打印每个张量的 `shape`，证明其维度闭合；
- 随机种子固定；
- 把结果与 `torch.autograd` 对比：每个参数相对误差小于 `1e-4`；
- 之后用学习率 `lr=0.1` 做一次纯 PyTorch 参数更新，不调用优化器；
- 验证 loss 下降，并解释 `optimizer.step()` 到底做了什么。

### 逐步推演

令 batch 为 `N`。前向各步：

```text
X   : (N, D)
W1  : (D, H)
b1  : (H,)
G1  = X @ W1 + b1        → (N, H)
H_  = ReLU(G1)           → (N, H)
W2  : (H, C)
b2  : (C,)
S   = H_ @ W2 + b2       → (N, C)
```

这里 `@` 是矩阵乘法。最后一层不手动 softmax；交叉熵内部使用 logits，真实类别用类别索引。

反向要从标量损失开始。设 `P` 是 softmax 概率，`Y` 是 one-hot 目标，对交叉熵先得到：

\[
\frac{\partial L}{\partial s_{i,j}}
=
\frac{1}{N}\left(p_{i,j}-y_{i,j}\right)
\]

批平均会引入 `1/N`；如果 `reduction="sum"`，就没有这个因子。这是最容易漏的一步。

随后：

\[
dW_2 = H^{\top} dS,\quad db_2 = dS^{\top}\mathbf{1}_N
\]

\[
dH = dS W_2^{\top}
\]

ReLU 的导数在输入大于 0 时为 1、否则为 0：

\[
dG_1 = dH \odot \mathbb{1}_{G_1>0}
\]

最后：

\[
dW_1 = X^{\top} dG_1,\quad db_1 = dG_1^{\top}\mathbf{1}_N
\]

关键检查是维度是否闭合：

```text
dS    : (N, C)   dW2 : (H, C)   → H^T(N,H)^T @ dS(N,C) = (H,C)
db2   : (C,)     ← dS 沿 N 求和
dH    : (N, H)   ← dS(N,C) @ W2^T(C,H)
dG1   : (N, H)   ← 与 dH 逐元素相乘
dW1   : (D, H)   ← X^T(D,N) @ dG1(N,H)
db1   : (H,)     ← dG1 沿 N 求和
```

如果某一步出现 `(N,H) @ (H,C)` 却得到 `(N,D)`，就要停下检查转置，而不是继续运行。矩阵求导在这里不是抽象公式，而是“谁乘谁、哪一个维必须对齐”。

### 参考实现

```python
import torch
import torch.nn.functional as F

torch.manual_seed(0)


def softmax_cross_entropy_grad(logits: torch.Tensor, target: torch.Tensor):
    # 稳定 log-softmax
    z = logits - logits.max(dim=1, keepdim=True).values
    e = torch.exp(z)
    p = e / e.sum(dim=1, keepdim=True)
    n = logits.shape[0]

    one_hot = torch.zeros_like(p)
    one_hot.scatter_(1, target.unsqueeze(1), 1.0)

    dlogits = (p - one_hot) / n  # 对应 mean reduction
    loss = -torch.log(torch.gather(p, 1, target.unsqueeze(1))).mean()
    return loss, dlogits


def manual_mlp(X, y, D=5, H=4, C=2, lr=0.1, steps=20):
    W1 = torch.randn(D, H, requires_grad=True)
    b1 = torch.zeros(H, requires_grad=True)
    W2 = torch.randn(H, C, requires_grad=True)
    b2 = torch.zeros(C, requires_grad=True)

    history = []
    for _ in range(steps):
        G1 = X @ W1 + b1
        H_ = torch.relu(G1)
        S = H_ @ W2 + b2
        loss, dS = softmax_cross_entropy_grad(S, y)

        dW2 = H_.t() @ dS
        db2 = dS.sum(dim=0)
        dH = dS @ W2.t()
        dG1 = dH * (G1 > 0).float()
        dW1 = X.t() @ dG1
        db1 = dG1.sum(dim=0)

        with torch.no_grad():
            W1 -= lr * W1.grad if W1.grad is not None else lr * dW1
            b1 -= lr * b1.grad if b1.grad is not None else lr * db1
            W2 -= lr * W2.grad if W2.grad is not None else lr * dW2
            b2 -= lr * b2.grad if b2.grad is not None else lr * db2

        history.append(loss.item())
        assert dW1.shape == W1.shape
        assert db1.shape == b1.shape
        assert dW2.shape == W2.shape
        assert db2.shape == b2.shape

    return history


def compare_with_autograd(X, y, D=5, H=4, C=2):
    W1 = torch.randn(D, H, requires_grad=True)
    b1 = torch.zeros(H, requires_grad=True)
    W2 = torch.randn(H, C, requires_grad=True)
    b2 = torch.zeros(C, requires_grad=True)

    G1 = X @ W1 + b1
    H_ = torch.relu(G1)
    S = H_ @ W2 + b2
    loss = F.cross_entropy(S, y)
    loss.backward()

    _, dS = softmax_cross_entropy_grad(S, y)
    dW2 = H_.t() @ dS
    db2 = dS.sum(dim=0)
    dH = dS @ W2.t()
    dG1 = dH * (G1 > 0).float()
    dW1 = X.t() @ dG1
    db1 = dG1.sum(dim=0)

    # autograd 已反向一次，避免在同一图上再次 backward
    assert W1.grad is not None and W2.grad is not None
    assert torch.allclose(dW1, W1.grad, atol=1e-5, rtol=1e-4)
    assert torch.allclose(db1, b1.grad, atol=1e-5, rtol=1e-4)
    assert torch.allclose(dW2, W2.grad, atol=1e-5, rtol=1e-4)
    assert torch.allclose(db2, b2.grad, atol=1e-5, rtol=1e-4)


def demo():
    torch.manual_seed(0)
    X = torch.randn(3, 5)
    y = torch.tensor([0, 1, 0])

    compare_with_autograd(X, y)
    history = manual_mlp(X, y)
    assert history[-1] < history[0]
    print("Day 02 C passed:", history[0], "->", history[-1])


if __name__ == "__main__":
    demo()
```

### 常见错误与调试方法

| 错误/现象 | 根因 | 处理 |
|---|---|---|
| `shape mismatch` | 矩阵乘方向错误 | 写“前维=后维、剩余维=结果”的小表 |
| 梯度差一个转置 | 把 `X @ G1` 写成 `G1 @ X` | 用 `(D,N)` 与 `(N,H)` 对齐维 |
| 只对一个样本成立 | 漏掉 batch 求和/平均 | `dW2=H_.t() @ dS` 中 batch 维自动收缩 |
| loss 不降 | 学习率太小、符号错、漏 ReLU 梯度门控 | 先打印 `lr * grad` 数量级；确认 `W -= lr*grad` |
| loss 立刻变成 nan | logits 过大、学习率过大 | 缩小初始化/学习率；检查 softmax 稳定性 |
| `W1.grad is None` | 该张量未参与当前图，或已 detached | 检查 `requires_grad` 和 `.grad_fn` |
| 第二次 backward 报错 | 图已释放 | 调试才用 `retain_graph=True`，不用于常规训练 |
| 优化器没效果 | `zero_grad()` 位置错、更新后图没重建 | 每个 step 前清零，forward 后重新建图 |
| `optimizer.step()` 报错 | state 与参数不匹配 | 先把参数全部交给 optimizer，再 load_state_dict |

官方资料强调，梯度会累加到张量的 `.grad`；因此若缺失清零，多个 batch 的梯度会叠加[1]。CS231n 对训练过程的工程建议也包括在自定义操作时做梯度检查[3]。今天这题就是最小梯度检查：自动求导是参考答案，手算是你的解释。

### 验收标准

```python
def test_exercise_c():
    torch.manual_seed(0)
    X = torch.randn(4, 6)
    y = torch.tensor([0, 2, 1, 0])

    compare_with_autograd(X, y, D=6, H=5, C=3)

    history = manual_mlp(X, y, D=6, H=5, C=3, lr=0.1, steps=30)
    assert len(history) == 30
    assert history[-1] < history[0]
    assert all(torch.isfinite(torch.tensor(v)) for v in history)
```

### 与 4D Occupancy 的关联

卷积层只是把“矩阵乘法”换成“局部连接、参数共享的滤波运算”；链式法则仍然是逐算子反向。第4周面对 3D 卷积时，你不需要现场推导所有 CUDA 算子，但要能回答：

- 前向输入是什么形状、输出是什么形状；
- 哪个张量需要梯度；
- 若 loss 没有下降，是数据、前向、loss、梯度还是优化器的问题；
- 若梯度为 `None`，哪一步把图切断了；
- 若梯度爆炸，是先剪裁、降学习率，还是检查损失是否缺少平均/归一化。

第5周 ablation 会同时改变主干、时序融合和损失权重。若你不理解梯度更新，很容易把“模型变好”错误归因于某个模块，而真正原因可能是学习率、batch 统计、随机种子或梯度累积步数变化。第6周云端训练更要理解：优化器状态、混合精度、梯度裁剪和分布式同步都位于“loss→grad→update”这一链条上，不是训练之外的魔法设置。

## “第一次必得 nan”的刻意练习：让失败成为证据

**不要让学习者直接看稳定答案，而要让他先承担一次可控制的计算失败。** 推荐把练习 B 拆成四步，每步都在终端运行，不许复制粘贴整套答案。

### 第一步：先背“朴素公式”，不谈稳定性

要求手写：

```python
def naive_ce(logits, target):
    probs = torch.exp(logits) / torch.exp(logits).sum(dim=1, keepdim=True)
    return -torch.log(
        probs.gather(dim=1, index=target.unsqueeze(1)).squeeze(1)
    ).mean()
```

随后口述：

1. 为什么需要 `sum(dim=1, keepdim=True)`？
2. 为什么要在 `target` 最后增加一维？
3. 为什么取真实类别要用 `gather` 而不是 `probs[:, target]`？

这三问不是浪费时间。它们把“公式”翻译成“维度操作”，也暴露是否理解 batch 维。

### 第二步：用自己设计的最小输入制造失败

给学习者一个故意不明显的输入：

```python
torch.manual_seed(0)
logits = torch.randn(2, 3) * 10
target = torch.tensor([0, 2])
```

再给一个显然会失败的反例：

```python
logits = torch.tensor([[1000.0, 0.0]])
target = torch.tensor([0])
print(torch.softmax(logits, dim=1))
print(naive_ce(logits, target))
```

观察到 softmax 概率可能正常、朴素损失却变成 `inf/nan`。这里可以问：“为什么概率还能算、取对数却坏了？”答案是中间计算与最终概率可能在数值路径上不同；稳定实现控制的是整个 log-sum-exp 路径，而不是只让最终概率好看。

### 第三步：只给两条线索，让他自己发现修复

不要直接给 `log_softmax`。先问：

> 如果分子和分母同时除以同一个正数，分数会改变吗？

再给：

> 什么样的正数最容易算、又能保证所有指数项不超过 1？

学习者若能写出：

\[
\frac{\exp(z_j)}{\sum_k\exp(z_k)}
=
\frac{\exp(z_j)/\exp(M)}{\sum_k\exp(z_k)/\exp(M)}
\]

其中 `M = max(z)`，就说明他已经把稳定性理解为代数恒等变换，而不是 API 口诀。

### 第四步：把实验写成“证据记录”，而不是结论摘录

```python
records = []
for case in [
    torch.tensor([[5.0, 1.0]]),
    torch.tensor([[1000.0, 0.0]]),
    torch.tensor([[-1000.0, 0.0]]),
]:
    records.append({
        "case": case,
        "naive_finite": torch.isfinite(naive_ce(case, torch.tensor([0]))).item(),
        "stable_finite": torch.isfinite(my_cross_entropy(case, torch.tensor([0]))).item(),
    })
```

这一步尤其适合你的学习风格：**看见报错不是终点，把报错变成表格才算完成。** 它也会成为第4周处理损失异常时的模板：复现最小反例、记录有限性、二分定位、再修复。

## 完成标准必须写成可观察行为，而不是“我大概懂了”

**Day 02 结束的标准是：不查资料也能写出四个流程，且机器断言通过。** 建议进行十分钟闭卷验收，完全不打开答案和文档。

1. 给定一个 `shape=(B,T,H,W,C)` 的张量，能在 60 秒内写出以下结果形状：取第2个时间步、第5~15高度、第20~40宽度、通道 `[0,2]`；结果为 `(10,20,2)`。
2. 能不查资料写出 `softmax` 和交叉熵公式，并说明为何要减 max。
3. 能写出 `log_softmax+nll` 的完整最小实现，并在 `1e-6` 内与官方 API 对齐。
4. 能写出两层 MLP 的前向、损失、四个参数梯度和一次参数更新。
5. 能在白板说明：`requires_grad`、计算图、`backward()`、梯度清零、优化器更新之间的关系。
6. 三个练习的 `pytest`/脚本全部通过；失败用例不是靠删掉 `assert`，而是修改代码后通过。

若只能完成 1–3，但说不清链式法则，不算通过；若能把 MLP 推导出来，但 A 的形状仍然靠猜，也不算通过。今天的成功标准是**四条链路都通**。

## 本地环境命令：不要混用全局 Python 与 .venv

**环境复现必须固定解释器和锁定文件，不能只说“我装了 PyTorch”。** 在仓库根目录执行：

```powershell
python --version
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install pytest
python -c "import torch, pytest; print(torch.__version__, pytest.__version__)"
pytest day02 -q
```

如果项目已有 `requirements.lock`，不要绕过它：

```powershell
python -m pip install -r requirements.lock
```

版本字符串必须写到 `2.14.1+cpu`；只写“PyTorch 2.x”不足以复现第6周云端实验。Colab T4 应作为性能测试环境，不应成为本地练习的唯一运行环境；否则未来网络、镜像或运行时变化会让你误以为“代码出了问题”。

## 目录结构与可直接运行的验收脚本

**把每个练习独立成文件，才能让失败定位到一行。** 建议仓库结构：

```text
worldmodel-4d-occupancy/
├─ .venv/
├─ requirements.lock
├─ day02/
│  ├─ __init__.py
│  ├─ exercise_a_indexing.py
│  ├─ exercise_b_cross_entropy.py
│  └─ exercise_c_backprop.py
├─ tests/
│  └─ test_day02.py
├─ docs/
│  ├─ conventions.md
│  ├─ day01.md
│  └─ day02.md
└─ README.md
```

`tests/test_day02.py` 应导入三题的测试函数：

```python
from day02.exercise_a_indexing import test_exercise_a
from day02.exercise_b_cross_entropy import test_exercise_b
from day02.exercise_c_backprop import test_exercise_c


def test_a():
    test_exercise_a()


def test_b():
    test_exercise_b()


def test_c():
    test_exercise_c()
```

运行命令固定为：

```powershell
.\.venv\Scripts\Activate.ps1
pytest tests/test_day02.py -v
```

这样“完成”不是主观感觉，而是 `3 passed`。

## Day 02 的 git 提交清单

**提交的是“可观察证据”，不是整堆试错文件。** 当天至少分两个有意义的提交，不要最后一次性 `git add .`。

### 提交 1：`day02: add gap self-assessment and three minimal exercises`

```text
day02/
├─ exercise_a_indexing.py
├─ exercise_b_cross_entropy.py
├─ exercise_c_backprop.py
└─ gap_checklist.md
docs/conventions.md
tests/test_day02.py
```

提交说明建议：

```text
day02: add gap self-assessment and three minimal exercises

- Add Python/tensor/autograd/neural-network gap checklist.
- Implement BEV window slicing, boolean/gather/index_select and view/reshape checks.
- Implement numerically stable log-softmax + NLL cross entropy.
- Implement a 2-layer MLP with manual forward, backward, and parameter update.
- Add deterministic tests covering all three exercises.
```

### 提交 2：`docs(day02): record failures, fixes and post-mortem`

```text
docs/day02.md
docs/conventions.md
tests/test_day02.py
day02/
```

提交说明建议：

```text
docs(day02): record failures, fixes and post-mortem

- Document the manual cross-entropy overflow reproduction.
- Record shape derivations for 4D BEV tensors.
- Explain the chain-rule dimensions of the 2-layer MLP.
- List remaining gaps and Day 03 starting checkpoint.
```

当天不应提交：

- `__pycache__/`；
- `.venv/`；
- `.ipynb_checkpoints/`；
- 编辑器临时文件；
- 原始错误信息截图；
- 直接下载的大模型权重；
- `final_loss=0.0366` 的误读或没有代码佐证的数字。

如果 Day 02 没有完全通过，不要伪装成通过。建议提交 `day02-WIP: failed at step X, reproducible minimal example added`，并在 `docs/day02.md` 中写清卡点；这本身就是科研式工程记录。

## docs/day02.md 填写模板

**模板的作用是逼你写因果，而不是写课堂笔记。**

```markdown
# Day 02 复盘

## 日期与环境
- 日期：2026-10-23
- Windows / Python：Windows 11 / Python 3.10
- 本地 PyTorch：2.14.1+cpu
- 云端：Colab T4（若当天使用）
- 随机种子：`torch.manual_seed(0)`
- 完成状态：通过 / 部分通过 / 未通过

## 一句话结论
今天我发现，我之前能跑 TinyCNN 只说明调用顺序正确，并不能证明我会预测张量形状和梯度维度。

## 差距清单自评
| 维度 | 0 不会 | 1 能照做 | 2 能迁移 | 证据 |
|---|---:|---:|---:|---|
| Python 作用域/可变对象 |  |  |  |  |
| 张量 shape 与 indexing |  |  |  |  |
| bool/gather/view/reshape |  |  |  |  |
| log_softmax 与 NLL |  |  |  |  |
| requires_grad 与计算图 |  |  |  |  |
| 梯度清零与累加 |  |  |  |  |
| 二层 MLP 链式法则 |  |  |  |  |
| 参数更新流程 |  |  |  |  |

最意外的一条：
最危险的一条：

## 练习 A
- 预测结果 shape：
- 实际结果 shape：
- bool 索引为何是一维：
- `view` 与 `reshape` 的差别：
- 与未来 `(B,T,H,W,C)` 约定的关联：
- 验收：`pytest tests/test_day02.py::test_a`

## 练习 B：手算 nan 实验
- 朴素实现为何溢出：
- 我第一次观察到的数值：
- 为何可以减去同一批最大值：
- 手算小例子：logits=[[1,2]]，target=[1]
- 与官方 API 的最大绝对差：
- 若只差 1e-5，我会检查：
- 验收：`pytest tests/test_day02.py::test_b`

## 练习 C：二层 MLP 手算梯度
| 张量 | shape | 我的错误 |
|---|---|---|
| X |  |  |
| W1 |  |  |
| G1 |  |  |
| H |  |  |
| W2 |  |  |
| S |  |  |
| dS |  |  |
| dW2 |  |  |
| db2 |  |  |
| dH |  |  |
| dG1 |  |  |
| dW1 |  |  |
| db1 |  |  |

- 相对误差：
- 首次 loss / 最终 loss：
- `W -= lr*grad` 必须在什么上下文执行：
- 与 torch.autograd 的差别：
- 验收：`pytest tests/test_day02.py::test_c`

## 我今天真正学会的三句话
1. 
2. 
3. 

## 我仍说不清的三件事
1. 
2. 
3. 

## 明日起点
- Day 03 第一件事：
- 今天保留下来的最小失败案例：
- 30 分钟必做项：
```

## 常见的“看起来会了其实没会”陷阱

**这五类幻觉会在第4周后集中爆发，因此今天必须提前识别。**

1. **“我会索引”的幻觉**：只记得 `x[2,10:20,30:40]`，却不解释第0维为何消失。检验方法是随机改一个维度，让你预测新 shape。
2. **“我懂 loss”的幻觉**：能调用 `F.cross_entropy`，但把 `softmax(logits)` 再传进去，或把 float target 当类别索引。检验方法是手写 NLL。
3. **“我会反向传播”的幻觉**：能写 `loss.backward()`，但不知道为什么要从标量开始、梯度为何累加。检验方法是关掉 autograd，手算一次。
4. **“代码能跑就是完成”的幻觉**：缺少 `assert`，只凭 `print` 判断。检验方法是删除所有打印语句，让 `pytest` 成为唯一裁判。
5. **“我在做 4D Occupancy”的幻觉**：过早下载论文代码、复制 BEVFormer，却连 4D 张量的索引都说不清。正确顺序是先把 2D/3D 基础零件做成本地可验证工具，再接入论文结构。

CS231n 对网络训练与梯度检查的建议，本质上是把“模型直觉”变成“可验证计算”[3]。你的项目最终不是要贴出一张漂亮的损失曲线，而是要能在第5周说出：哪个模块改了什么、损失为何变化、梯度是否流动、结果是否由随机性造成。今天这 180 分钟，就是这套能力的最小训练场。

## 引用来源

[1] https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html
> Tensors, Function objects, and Autograd connect and build a non-cyclic graph, that encodes a complete history of computation.

[2] https://pytorch.org/docs/stable/generated/torch.nn.functional.cross_entropy.html
> This criterion computes the cross entropy loss between input logits and target.

[3] https://cs231n.github.io/neural-networks-3/
> Gradient checks take a while to compute, so should only be performed every once in a while.

[4] https://www.docin.com/p-4680611053.html
> 自动梯度示例……out = x.pow(2).sum()……out.backward()

[5] https://www.doc88.com/p-5466378570835.html
> 从左到右，开头的字符索引是0，第二个字符索引是1……最后一个字符索引是-1。

[6] https://www.math.pku.edu.cn/teachers/qiuzy/computing/courseware/lecture08-prog3.pdf
> 弄清在一个程序里，哪些名字出现代表同一个变量，哪些虽然名字相同，但表示的是不同的变量。

[7] https://www.docin.com/p-4404139786.html
> 对于一个将要被 view 的 Tensor，新的 size 必须与原来的 size 与 stride 兼容。

[8] https://www.zztrc.edu.cn/__local/0/4B/28/F5ECBC35CF149FB25C7D220D053_F29A52FC_793DC.pdf
> 张量基础（一）：创建、属性、索引……掌握张量的创建方法、属性查看、索引与切片操作。

[9] https://view.inews.qq.com/a/20260705A02SDX00
> 从技术路线到产业落地，各家走出了不同的路。

[10] https://baike.baidu.com/item/BEV/68887973
> 3D Occupancy开始受到重视，世界模型成为新的研究热点。
