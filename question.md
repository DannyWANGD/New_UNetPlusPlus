# CLAUDE.md 中“改动一”的问题清单

本文将 `CLAUDE.md` 中“改动一：跳跃路径 H 替换为 MLKA”部分的问题整理为 Q1、Q2... 的形式，便于后续逐条修订。

## Q1：全节点维度表是否正确？

**问题：不正确。**

`CLAUDE.md` 中把多个节点的输出通道写成了下一层较小通道，例如：

```text
x1_1: Conv3x3(60 -> 30)，输出 30
x2_1: Conv3x3(120 -> 60)，输出 60
x3_1: Conv3x3(240 -> 120)，输出 120
x4_1: Conv3x3(480 -> 240)，输出 240
```

但当前 `nnUNetPlusPlusTrainerV2.py` 初始化网络时使用：

```python
convolutional_pooling=True
convolutional_upsampling=True
```

在这个设置下，`create_nest()` 中每个节点的第二步输出应保持为 `nfeatures_from_skip`，也就是当前行的通道数。

**正确理解：**

```text
x^{i,j}: concat 输入通道 = (j + 1) * C_i
MLKA 输出通道 = C_i
```

例如：

```text
x1_1: concat 输入 120，输出 60
x2_1: concat 输入 240，输出 120
x3_1: concat 输入 480，输出 240
x4_1: concat 输入 960，输出 480
```

**建议：**

重写 `CLAUDE.md` 中的全节点维度表，按当前 `convolutional_upsampling=True` 的真实行为填写。

## Q2：`MLKABlock(C_i, C_i)` 是否适用于所有 UNet++ 配置？

**问题：不一定。**

文档中默认将第二步统一替换为：

```text
MLKABlock(C_i, C_i)
```

这个假设只在当前 `convolutional_upsampling=True` 的配置下成立。

如果以后切换为 `convolutional_upsampling=False`，原始第二步可能是：

```text
C_i -> C_{i-1}
```

此时只使用 `MLKABlock(channels=C_i)` 会导致输出通道错误，后续节点无法拼接。

**建议：**

二选一：

```text
方案 A：明确说明当前改动一仅支持 convolutional_upsampling=True。
方案 B：将 MLKABlock 设计为 MLKABlock(in_channels, out_channels)，末尾用 1x1 projection 适配输出通道。
```

## Q3：`MLKABlock` 和 `GroupGLKA` 的封装关系是否清楚？

**问题：不清楚，且存在前后矛盾。**

文档前面描述的 `MLKABlock` 内部结构是：

```text
LayerNorm -> 1x1 Conv(C_i -> 2C_i) -> chunk -> 三组 LKA -> 1x1 Conv(C_i -> C_i) -> residual
```

这实际上已经接近 `MAN_arch.py` 中 `GroupGLKA` 的内部结构。

但文档后面又写：

```text
MLKABlock 包装 LayerNorm -> 1x1(C -> 2C) -> GroupGLKA(2C) -> 1x1(C -> C) -> Residual
```

这个说法容易导致“双重扩展”，因为 `GroupGLKA` 内部本来就有：

```python
self.proj_first = nn.Conv2d(n_feats, 2 * n_feats, 1)
```

**建议：**

明确采用一种实现：

```text
方案 A：MLKABlock = 改造版 GroupGLKA(channels)，直接把 GroupGLKA 作为精炼模块。
方案 B：不套用 GroupGLKA，单独实现 MLKACore，由 MLKABlock 自己负责扩展、分组、门控、投影和残差。
```

不要写成 `1x1(C -> 2C) -> GroupGLKA(2C)`，否则结构会变得不清晰。

## Q4：文档中的 sigmoid gate 是否来自 MAN 原始实现？

**问题：不是。**

`CLAUDE.md` 中描述三组分支时写了：

```text
DW3x3 -> sigmoid
DW5x5 -> sigmoid
DW7x7 -> sigmoid
```

但 `MAN_arch.py` 中的 `GroupGLKA` 实际写法是：

```python
a = torch.cat([
    self.LKA3(a_1) * self.X3(a_1),
    self.LKA5(a_2) * self.X5(a_2),
    self.LKA7(a_3) * self.X7(a_3)
], dim=1)
```

其中 `X3/X5/X7` 是 depthwise conv，并没有 sigmoid。

**建议：**

在文档里明确区分：

```text
MAN 原始 GroupGLKA：LKA branch * depthwise branch
改进版 sigmoid gate：LKA branch * sigmoid(depthwise branch)
```

如果决定加入 sigmoid，需要说明这是你的二次设计，不是直接复用 MAN 原实现。

## Q5：默认归一化层写成 BatchNorm 是否合理？

**问题：不够准确。**

文档中多处写：

```text
默认使用 BatchNorm，与 nnU-Net 一致
```

但当前 `nnUNetPlusPlusTrainerV2.initialize_network()` 中实际是：

```python
if self.threeD:
    norm_op = nn.InstanceNorm3d
else:
    norm_op = nn.InstanceNorm2d
```

也就是说，当前 trainer 默认使用 InstanceNorm，而不是 BatchNorm。

**建议：**

将默认归一化策略改为：

```text
mlka_norm='instance'
```

并支持消融：

```text
instance: 与当前 nnU-Net trainer 保持一致
layer: 保持 MAN 原始风格
batch: 作为额外对照实验
```

## Q6：`MLKABlock` 是否需要 `output_channels` 属性？

**问题：需要。**

原始 `StackedConvLayers` 中有：

```python
self.output_channels = output_feature_channels
```

而 `Generic_UNetPlusPlus.__init__()` 创建深监督输出头时依赖：

```python
self.loc0[-1][-1].output_channels
self.loc1[-1][-1].output_channels
self.loc2[-1][-1].output_channels
self.loc3[-1][-1].output_channels
self.loc4[-1][-1].output_channels
```

如果第二步替换为 `MLKABlock`，但 `MLKABlock` 没有 `output_channels`，构建 `seg_outputs` 时会报错。

**建议：**

在 `MLKABlock.__init__()` 中加入：

```python
self.output_channels = out_channels
```

如果当前只支持 `channels -> channels`，也至少需要：

```python
self.output_channels = channels
```

## Q7：改动一是否默认适用于 3D fullres？

**问题：不适用。**

`MAN_arch.py` 中的模块全部基于：

```python
nn.Conv2d
```

但 3D fullres 网络内部使用：

```python
nn.Conv3d
```

因此不能直接把 `MAN_arch.py` 的 `GroupGLKA` 或 `MLKABlock` 放进 3D UNet++。

**建议：**

在文档中明确：

```text
当前改动一优先针对 2D 遥感分割。
若要支持 3D fullres，需要额外实现 Conv3d 版 MLKA。
```

3D 版本至少需要适配：

```text
Conv2d -> Conv3d
InstanceNorm2d -> InstanceNorm3d
kernel_size: 7 -> (3, 7, 7) 或 (7, 7, 7)
输入维度: B,C,H,W -> B,C,D,H,W
```

## Q8：正确的节点维度表应该如何写？

以当前 trainer 默认设置为准：

```text
base_num_features = 30
max_features = 480
num_pool = 5
convolutional_upsampling = True
```

正确表格如下：

| 节点 | concat 输入通道 | 降维后通道 | MLKA 输出通道 |
|------|:--------------:|:----------:|:------------:|
| x0_1 | 60 | 30 | 30 |
| x1_1 | 120 | 60 | 60 |
| x0_2 | 90 | 30 | 30 |
| x2_1 | 240 | 120 | 120 |
| x1_2 | 180 | 60 | 60 |
| x0_3 | 120 | 30 | 30 |
| x3_1 | 480 | 240 | 240 |
| x2_2 | 360 | 120 | 120 |
| x1_3 | 240 | 60 | 60 |
| x0_4 | 150 | 30 | 30 |
| x4_1 | 960 | 480 | 480 |
| x3_2 | 720 | 240 | 240 |
| x2_3 | 480 | 120 | 120 |
| x1_4 | 300 | 60 | 60 |
| x0_5 | 180 | 30 | 30 |

## Q9：更严谨的改动一公式应该如何表述？

建议不要简单写成“把 H 替换成 MLKA”，而是写成：

```text
H_new = ConvReduce(concat_channels -> node_channels)
      + MLKARefine(node_channels -> node_channels)
```

其中：

```text
concat_channels = (j + 1) * C_i
node_channels = C_i
```

这样能避免误以为 MLKA 直接处理拼接后的大通道输入。

**建议补充说明：**

```text
当前版本优先针对 2D 遥感分割和 convolutional_upsampling=True 的 UNet++ 实现。
若要兼容 3D fullres 或 convolutional_upsampling=False，需要额外实现 Conv3d 版 MLKA 或 out_channels projection。
```

## Q10：只给 `Generic_UNetPlusPlus` 增加 `use_mlka` 参数是否足够？

**问题：不够。**

即使在 `Generic_UNetPlusPlus` 中增加：

```python
use_mlka=False
```

默认也不会启用 MLKA。当前 trainer 初始化网络时没有传入 `use_mlka=True`。

**建议：**

新建一个独立 trainer，例如：

```text
nnUNetPlusPlusTrainerV2_MLKA
```

在该 trainer 的 `initialize_network()` 中显式传入：

```python
use_mlka=True
mlka_groups=3
mlka_norm='instance'
```

这样可以保留原始 `nnUNetPlusPlusTrainerV2` 作为 baseline，同时让改进模型有独立入口，便于后续消融实验。

