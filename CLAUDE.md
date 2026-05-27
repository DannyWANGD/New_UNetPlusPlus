# CLAUDE.md — New_UNetPlusPlus 改进项目

## 项目概述

本项目基于 nnU-Net 框架的 UNet++（`generic_UNetPlusPlus.py`），将 MAN 论文（arXiv:2209.14145）中的多尺度大核注意力（MLKA）与门控空间注意力单元（GSAU）引入 UNet++ 架构，用于遥感语义分割。核心目标：在最小改动原有架构的前提下，增强模型的多尺度感知与长程依赖建模能力，同时保持 UNet++ 的深度监督剪枝优势。

## 关键参考文件

| 文件 | 用途 |
|------|------|
| `pytorch/nnunet/network_architecture/generic_UNetPlusPlus.py` | **主修改目标** — UNet++ 网络定义 |
| `MAN_arch.py` | **模块来源** — 包含 LKA / MLKA_Ablation / GroupGLKA / SGAB / LayerNorm 等可复用模块 |
| `pytorch/nnunet/training/network_training/nnUNetPlusPlusTrainerV2.py` | **训练器** — 网络实例化入口，决定 conv_op / norm_op / 上采样模式 |
| `基于多尺度大核注意力的 UNet++ 改进：理论、方法与实验.md` | 理论方案文档（Section 6 定义改动一、二） |
| `UNetPlusPlus_Core_Architecture.md` | UNet++ 数学推导与架构详解 |
| `UNetPlusPlus_Expert_Guide.md` | nnU-Net 训练管线与开发指南 |
| `MLKA_GSAU_改进方案评估与建议.md` | 改动方案的科学性评估与风险分析 |

---

## 改动一：跳跃路径 H 替换为 MLKA（详细方案）

### 1. 设计原则与公式

不直接把 `GroupGLKA(n_feats)` 塞进 UNet++，因为 `GroupGLKA` 要求 **C_in = C_out**，而 UNet++ 节点的 H 是 **多源拼接 (j+1 路) → 卷积融合 → 单路输出**。采用两步解耦：

```
H_new = ConvReduce(concat_channels → C_i) + MLKARefine(C_i → C_i)

其中：
  concat_channels = (j + 1) × C_i    （拼接后总通道）
  C_i             = 当前层编码器通道  （nnU-Net 中即 nfeatures_from_skip）
```

保留第一步的降维融合（Conv3×3），仅将第二步的 3×3 精炼卷积替换为 MLKA。MLKA 职责单一：输入 C_i，输出 C_i，做多尺度空间注意力增强。forward 中所有 `self.loc*[idx](...)` 调用方式**完全不变**。

**适用范围（重要）**：当前改动一仅针对 `convolutional_upsampling=True` 的 2D UNet++（与 `nnUNetPlusPlusTrainerV2` 默认配置一致）。如需支持 `convolutional_upsampling=False` 或 3D，见第 8 节说明。

### 2. 当前 H 的结构（改造前）

H 由 `create_nest()` 方法生成（[generic_UNetPlusPlus.py:466-473](pytorch/nnunet/network_architecture/generic_UNetPlusPlus.py#L466-L473)），每个节点是一个 `nn.Sequential`。

当前 trainer 使用 `convolutional_upsampling=True`，因此所有节点的第二步输出均保持为 `nfeatures_from_skip`（即 C_i）：

```python
nn.Sequential(
    # 第一步：降维融合  (j+1)×C_i → C_i
    StackedConvLayers(
        in_ch  = n_features_after_tu_and_concat,  # = (j+1) × C_i
        out_ch = nfeatures_from_skip,              # = C_i
        num_convs = num_conv_per_stage - 1         # 默认 = 1
    ),
    # 第二步：特征精炼  C_i → C_i
    StackedConvLayers(
        in_ch  = nfeatures_from_skip,              # = C_i
        out_ch = nfeatures_from_skip,              # = C_i（conv_upsampling=True 时恒等）
        num_convs = 1
    )
)
```

每个 `StackedConvLayers(num_convs=1)` 展开为一层 `ConvDropoutNormNonlin`：**Conv3×3 → Dropout → InstanceNorm → LeakyReLU**。

### 3. H 改造成 MLKA 后的结构（改造后）

```python
nn.Sequential(
    # 第一步：降维融合（保持不变）
    StackedConvLayers(in_ch=(j+1)×C_i, out_ch=C_i, num_convs=1),
    # 第二步：MLKA 多尺度精炼（替换原来的 StackedConvLayers）
    MLKABlock(channels=C_i, num_groups=3)
)
```

`MLKABlock` 本质上是 `MAN_arch.py` 中 `GroupGLKA` 的轻量适配版，结构如下：

```
输入 [B, C_i, S, S]
  │
  ├─ Norm(C_i)                ← 默认 InstanceNorm2d；norm_type='layer' 时为 LayerNorm(channels_first)
  │     [B, C_i, S, S]
  │
  ├─ 1×1 Conv(C_i → 2×C_i)   ← 通道扩展（SimpleGate 机制：2×C_i 后 chunk 为二）
  │     [B, 2×C_i, S, S]
  │
  ├─ chunk(2, dim=1)          ← 拆分为 LKA 分支 a 和被调制分支 x
  │     a: [B, C_i, S, S]     x: [B, C_i, S, S]
  │
  ├─ 【a 分 3 组做多尺度 LKA】
  │     chunk(3, dim=1)
  │     a1: [B, C_i/3, S, S]     a2: [B, C_i/3, S, S]     a3: [B, C_i/3, S, S]
  │       │                         │                         │
  │     LKA3(a1)                 LKA5(a2)                 LKA7(a3)
  │     ├─ DW 3×3               ├─ DW 5×5               ├─ DW 7×7
  │     ├─ DW-dil 5×5,d=2       ├─ DW-dil 7×7,d=3       ├─ DW-dil 9×9,d=4
  │     └─ PW 1×1               └─ PW 1×1               └─ PW 1×1
  │       │                         │                         │
  │     × X3(a1)                 × X5(a2)                 × X7(a3)
  │     (DW 3×3,无激活)          (DW 5×5,无激活)          (DW 7×7,无激活)
  │       │                         │                         │
  │       └─────────── concat ─────────┘
  │                 a': [B, C_i, S, S]    ← 多尺度注意力特征
  │
  ├─ a' × x                   ← 逐元素调制（SimpleGate）
  │     [B, C_i, S, S]
  │
  ├─ 1×1 Conv(C_i → C_i)     ← 通道投影
  │     [B, C_i, S, S]
  │
  ├─ × scale + shortcut       ← 残差连接，scale 初始化为 0（恒等启动）
  │     [B, C_i, S, S]
  │
  输出 [B, C_i, S, S]
```

**关键说明**：上述结构即 `MAN_arch.py` 中 `GroupGLKA` 的原始结构。X3/X5/X7 是纯 depthwise conv，**无 sigmoid、无激活函数**——这是 MAN 论文的原设计，LKA branch 与 DWConv branch 直接逐元素相乘。MLKABlock 与 GroupGLKA 的关系是**适配而非包装**：在 `custom_modules/mlka.py` 中复制并适配 GroupGLKA，将内部归一化改为可配置（默认 InstanceNorm2d，与 nnU-Net 一致），并增加 `output_channels` 属性。

### 4. 全节点维度表

以下以当前 trainer 默认配置为准：

```
base_num_features = 30
max_num_features = 480
num_pool = 5
convolutional_upsampling = True
```

**编码器输出（conv_blocks_context）：**

| 索引 | 变量 | 输出通道 | 空间分辨率（相对） |
|:----:|------|:------:|:---:|
| 0 | x0_0 | 30 | H × W |
| 1 | x1_0 | 60 | H/2 × W/2 |
| 2 | x2_0 | 120 | H/4 × W/4 |
| 3 | x3_0 | 240 | H/8 × W/8 |
| 4 | x4_0 | 480 | H/16 × W/16 |
| 5 | x5_0 | 480 | H/32 × W/32（瓶颈） |

**每个跳跃节点的 H 维度（改动发生的位置）：**

| 节点 | 变量 | concat 输入通道 | 第一步：降维 | 第二步（改造后） | 输出通道 | 分辨率 |
|------|:---:|:---:|------|------|:---:|:---:|
| loc4[0] | x0_1 | 60 | Conv(60→30) | **MLKABlock(30)** | 30 | H×W |
| loc3[0] | x1_1 | 120 | Conv(120→60) | **MLKABlock(60)** | 60 | H/2×W/2 |
| loc3[1] | x0_2 | 90 | Conv(90→30) | **MLKABlock(30)** | 30 | H×W |
| loc2[0] | x2_1 | 240 | Conv(240→120) | **MLKABlock(120)** | 120 | H/4×W/4 |
| loc2[1] | x1_2 | 180 | Conv(180→60) | **MLKABlock(60)** | 60 | H/2×W/2 |
| loc2[2] | x0_3 | 120 | Conv(120→30) | **MLKABlock(30)** | 30 | H×W |
| loc1[0] | x3_1 | 480 | Conv(480→240) | **MLKABlock(240)** | 240 | H/8×W/8 |
| loc1[1] | x2_2 | 360 | Conv(360→120) | **MLKABlock(120)** | 120 | H/4×W/4 |
| loc1[2] | x1_3 | 240 | Conv(240→60) | **MLKABlock(60)** | 60 | H/2×W/2 |
| loc1[3] | x0_4 | 150 | Conv(150→30) | **MLKABlock(30)** | 30 | H×W |
| loc0[0] | x4_1 | 960 | Conv(960→480) | **MLKABlock(480)** | 480 | H/16×W/16 |
| loc0[1] | x3_2 | 720 | Conv(720→240) | **MLKABlock(240)** | 240 | H/8×W/8 |
| loc0[2] | x2_3 | 480 | Conv(480→120) | **MLKABlock(120)** | 120 | H/4×W/4 |
| loc0[3] | x1_4 | 300 | Conv(300→60) | **MLKABlock(60)** | 60 | H/2×W/2 |
| loc0[4] | x0_5 | 180 | Conv(180→30) | **MLKABlock(30)** | 30 | H×W |

**规律总结**：
- concat 输入通道 = `(j+1) × C_i`，其中 j 为列号（1~5）
- 第一步降维目标 = C_i（`nfeatures_from_skip`）
- 第二步 MLKA 输入/输出均为 C_i（保证 C_in = C_out）
- 输出通道始终 = C_i（`convolutional_upsampling=True` 的保证）

### 5. MLKABlock 内部维度示例（以 x0_2 为例，C_i=30）

```
MLKABlock(channels=30, num_groups=3)

输入 [B, 30, 256, 256]
  │
  ├─ InstanceNorm2d(30)           [B, 30, 256, 256]     参数: 60
  │
  ├─ Conv1×1(30 → 60)             [B, 60, 256, 256]     参数: 30×60+60 = 1860
  │
  ├─ chunk(2, dim=1) → a, x 各 [B, 30, 256, 256]
  │
  ├─ 【a 分 3 组，每组 10 通道】
  │   ┌─ a1 [10] ─ DW3 ─ DW-dil5,d=2 ─ PW(10→10) ─┐
  │   │                × X3(DW3, 无激活)            │
  │   ├─ a2 [10] ─ DW5 ─ DW-dil7,d=3 ─ PW(10→10) ─┤
  │   │                × X5(DW5, 无激活)            │
  │   └─ a3 [10] ─ DW7 ─ DW-dil9,d=4 ─ PW(10→10) ─┘
  │                    × X7(DW7, 无激活)
  │   concat → a': [B, 30, 256, 256]
  │   (3 组 LKA 参数 ≈ 470 + 870 + 1430 = 2770)
  │   (3 个 X* DWConv ≈ 100 + 260 + 500 = 860)
  │
  ├─ a' × x                        [B, 30, 256, 256]
  │
  ├─ Conv1×1(30 → 30)             [B, 30, 256, 256]     参数: 30×30+30 = 930
  │
  ├─ × scale [1,30,1,1] + shortcut  [B, 30, 256, 256]   参数: 30
  │
  输出 [B, 30, 256, 256]
```

**参数量合计（MLKABlock(30)）**：约 60 + 1860 + 2770 + 860 + 930 + 30 ≈ 6510。原始第二步（Conv3×3(30→30) + IN + LReLU）参数约 30×30×9 + 30 + 60 = 8190。**MLKA 参数仍更少，同时获得了 3 尺度感受野。**

### 6. 实施步骤（精确到函数与行号）

#### 步骤 1：创建 `custom_modules/mlka.py`

路径：`pytorch/nnunet/network_architecture/custom_modules/mlka.py`

从 `MAN_arch.py` 提取并适配以下类（**不修改 MAN_arch.py**）：

| 类名 | 来源 | 修改项 |
|------|------|--------|
| `LayerNorm` | MAN_arch.py:205-229 | 不变，保留 channels_first 支持 |
| `GroupGLKA` | MAN_arch.py:256-307 | 复制到 `custom_modules/mlka.py` 后适配：将 `self.norm` 从硬编码 LayerNorm 改为可配置（默认 InstanceNorm2d，与 nnU-Net trainer 一致） |
| `MLKABlock` | **新建** | 继承 GroupGLKA 逻辑，封装为统一接口，增加 `output_channels` 属性 |

`MLKABlock` 签名：

```python
class MLKABlock(nn.Module):
    """Multi-scale large kernel attention block. Adapts GroupGLKA for UNet++ nodes.
    
    Internally: Norm → 1×1(C→2C) → chunk → 3-group multi-scale LKA with DWConv gating → 1×1(C→C) → residual.
    The gating follows MAN's original design: LKA_branch × DWConv_branch (NO sigmoid).
    """
    def __init__(self, channels, num_groups=3, norm_type='instance'):
        # norm_type: 'instance' → InstanceNorm2d (default, matches nnU-Net trainer)
        #            'batch'    → BatchNorm2d
        #            'layer'    → LayerNorm(channels_first) (MAN original)
        # 当前实现固定采用 MAN GroupGLKA 的三分支结构，要求 channels % 3 == 0
        self.output_channels = channels  # 必须！seg_outputs 构建依赖此属性
```

**三分支通道检查**：

```python
if num_groups != 3:
    raise NotImplementedError("Current MLKABlock follows MAN GroupGLKA and supports exactly 3 groups.")
if channels % num_groups != 0:
    raise ValueError("MLKABlock requires channels divisible by 3 for GroupGLKA.")
if channels // num_groups < 8:
    raise ValueError("Each MLKA group should have at least 8 channels.")
```

当前默认通道 `30, 60, 120, 240, 480` 均满足三分支要求。若后续使用 `base_num_features=32` 等不能被 3 整除的配置，需要另行实现动态分支版 MLKA 或入口通道对齐层。

#### 步骤 2：修改 `generic_UNetPlusPlus.py` — 导入

在文件头部（L16 之后）添加：

```python
from nnunet.network_architecture.custom_modules.mlka import MLKABlock
```

#### 步骤 3：修改 `Generic_UNetPlusPlus.__init__()` — 参数

在 `__init__` 签名中增加参数（L192 附近）。**新增参数必须追加在 `seg_output_use_bias=False` 后面**，避免破坏 trainer 中已有的位置参数调用：

```python
def __init__(self, input_channels, base_num_features, num_classes, num_pool,
             num_conv_per_stage=2, ...,
             seg_output_use_bias=False,
             use_mlka=False,          # 新增：是否启用 MLKA
             mlka_groups=3,           # 新增：MLKA 分组数
             mlka_norm='instance'):   # 新增：MLKA 归一化类型（默认 InstanceNorm）
```

存储为实例属性：

```python
self.use_mlka = use_mlka
self.mlka_groups = mlka_groups
self.mlka_norm = mlka_norm
```

#### 步骤 4：修改 `create_nest()` — 方法签名

修改方法签名（L436）：

```python
def create_nest(self, z, num_pool, final_num_features, num_conv_per_stage,
                basic_block, transpconv, use_mlka=False, mlka_groups=3, mlka_norm='instance'):
```

#### 步骤 5：修改 `create_nest()` — 构建逻辑

将 L466-L473 的 `conv_blocks_localization.append(...)` 改为条件分支：

```python
# 第一步：降维融合（保持不变）
reduction = StackedConvLayers(
    n_features_after_tu_and_concat, nfeatures_from_skip,
    num_conv_per_stage - 1,
    self.conv_op, self.conv_kwargs,
    self.norm_op, self.norm_op_kwargs,
    self.dropout_op, self.dropout_op_kwargs,
    self.nonlin, self.nonlin_kwargs,
    basic_block=basic_block
)

# 第二步：特征精炼
if use_mlka:
    if not self.convolutional_upsampling:
        raise NotImplementedError("MLKA refinement currently requires convolutional_upsampling=True.")
    refinement = MLKABlock(
        channels=nfeatures_from_skip,   # = C_i
        num_groups=mlka_groups,
        norm_type=mlka_norm
    )
else:
    refinement = StackedConvLayers(
        nfeatures_from_skip, final_num_features, 1,  # 保持原始逻辑；当前 trainer 下 final_num_features == C_i
        self.conv_op, self.conv_kwargs,
        self.norm_op, self.norm_op_kwargs,
        self.dropout_op, self.dropout_op_kwargs,
        self.nonlin, self.nonlin_kwargs,
        basic_block=basic_block
    )

conv_blocks_localization.append(nn.Sequential(reduction, refinement))
```

#### 步骤 6：修改 `__init__()` — 传参给 create_nest

修改 5 处 `create_nest` 调用（L337-L346），统一追加参数：

```python
self.loc0, self.up0, _ = self.create_nest(
    0, num_pool, final_num_features, num_conv_per_stage,
    basic_block, transpconv,
    use_mlka=self.use_mlka, mlka_groups=self.mlka_groups, mlka_norm=self.mlka_norm
)
# loc1, loc2, loc3, loc4 同理
```

#### 步骤 7：创建独立 Trainer（必须）

在 `pytorch/nnunet/training/network_training/` 下新建 `nnUNetPlusPlusTrainerV2_MLKA.py`：

```python
from nnunet.training.network_training.nnUNetPlusPlusTrainerV2 import nnUNetPlusPlusTrainerV2

class nnUNetPlusPlusTrainerV2_MLKA(nnUNetPlusPlusTrainerV2):
    def initialize_network(self):
        # 同父类逻辑，仅在 Generic_UNetPlusPlus(...) 调用中追加：
        # use_mlka=True, mlka_groups=3, mlka_norm='instance'
        ...
        self.network = Generic_UNetPlusPlus(
            ...,
            use_mlka=True,
            mlka_groups=3,
            mlka_norm='instance'
        )
        ...
```

此 Trainer 独立于原始 `nnUNetPlusPlusTrainerV2`，保留后者作为 baseline，便于消融实验。

#### 步骤 8：验证

```python
# 测试脚本
common_kwargs = dict(
    conv_op=nn.Conv2d,
    norm_op=nn.InstanceNorm2d,
    norm_op_kwargs={'eps': 1e-5, 'affine': True},
    dropout_op=nn.Dropout2d,
    dropout_op_kwargs={'p': 0, 'inplace': True},
    nonlin=nn.LeakyReLU,
    nonlin_kwargs={'negative_slope': 1e-2, 'inplace': True},
    pool_op_kernel_sizes=[(2, 2)] * 5,
    conv_kernel_sizes=[(3, 3)] * 6,
    convolutional_pooling=True,
    convolutional_upsampling=True,
    weightInitializer=None,
)

model_orig = Generic_UNetPlusPlus(3, 30, 2, 5, use_mlka=False, **common_kwargs)
model_mlka = Generic_UNetPlusPlus(3, 30, 2, 5, use_mlka=True, **common_kwargs)

x = torch.randn(2, 3, 256, 256)
out_orig = model_orig(x)
out_mlka = model_mlka(x)
# 验证 1：use_mlka=True 前向/反向不报错，输出 shape 一致
# 验证 2：model_mlka.seg_outputs 正常构建（MLKABlock.output_channels 存在）
# 验证 3：use_mlka=False 输出与改造前数值一致（相同随机种子）
```

### 7. 改造前后对比

| 维度 | 改造前 | 改造后 |
|------|--------|--------|
| 每个节点的第二步 | Conv3×3(C_i→C_i) + IN + LReLU | MLKABlock(C_i, num_groups=3) |
| 感受野 | 固定 ~5×5 | 三尺度：~11×11 / ~23×23 / ~39×39 |
| 参数量（x0_2 节点） | ~8.2K | ~6.5K（仍更少） |
| 通道接口 | C_i → C_i | C_i → C_i |
| forward 调用方式 | `self.loc*[idx](cat(...))` | **不变** |
| seg_outputs 构建 | `self.loc*[-1][-1].output_channels` | **不变**（MLKABlock 暴露 output_channels） |
| 向后兼容 | — | `use_mlka=False` 完全恢复原行为 |

### 8. 改动一限制与实验边界

| 类别 | 当前边界 | 已验证行为 | 后续扩展方案 |
|------|----------|----------|--------------|
| **维度** | 仅支持 2D。`MAN_arch.py` 与当前 `MLKABlock` 均基于 `nn.Conv2d` / `InstanceNorm2d` | `conv_op=nn.Conv3d` 时主动抛出 `NotImplementedError` | 实现 Conv3d 版 MLKA：Conv2d→Conv3d, InstanceNorm2d→InstanceNorm3d, kernel 适配 3D（如 `(3,7,7)`） |
| **上采样方式** | 仅支持 `convolutional_upsampling=True` | `convolutional_upsampling=False` 时主动抛出 `NotImplementedError` | 若需支持非卷积上采样，修改为 `MLKABlock(in_ch, out_ch)`，末尾加 1×1 projection |
| **通道数** | `GroupGLKA` 固定 3 分支，要求所有 MLKA 节点通道 `C_i % 3 == 0` 且 `C_i / 3 >= 8`。当前实验限定 `base_num_features=30`，对应节点通道 `30, 60, 120, 240, 480` | `base_num_features=30` 通过；`base_num_features=32` 会报 `ValueError: channels must be divisible by num_groups` | 支持 32 等配置时，改为动态分支版 MLKA，或在 MLKA 前后增加通道对齐/投影层 |
| **拓扑深度** | 当前 `forward()` 是固定 5-pool UNet++ 显式拓扑 | 已按 `num_pool=5` 验证 15 个嵌套节点均为 `MLKABlock` | 如需 `num_pool != 5`，需重写 forward 为循环式拓扑或补齐显式节点 |
| **Trainer** | 原始 `nnUNetPlusPlusTrainerV2` 保持 baseline；MLKA 使用独立 `nnUNetPlusPlusTrainerV2_MLKA` | 新 trainer 显式传入 `use_mlka=True, mlka_groups=3, mlka_norm='instance'` | 后续消融可新增不同 trainer 或参数化 `mlka_norm` / `mlka_groups` |
| **归一化** | 默认 `mlka_norm='instance'`，与当前 2D trainer 的 `InstanceNorm2d` 对齐 | `instance` / `batch` / `layer` 的模块级前后向均通过 | 用独立实验比较不同归一化，不混入主实验 |
| **环境依赖** | 完整 nnU-Net 训练需要 `batchgenerators`、`SimpleITK`、`medpy` 等依赖；当前已在本地 `d2l` 环境补装这些依赖，并将 `numpy` 恢复到 `1.23.5` | 网络级构造和 trainer 初始化验证可直接导入；完整训练仍需设置 `PYTHONPATH` / nnU-Net 数据路径 | 在 GPU 实验环境中同步安装仓库依赖并配置数据路径 |

**已代入验证的网络案例**：

| 案例 | 结论 |
|------|------|
| `use_mlka=False` vs 当前 baseline，输入 `(1,3,64,64)`、`(2,3,96,128)` | 输出最大差值 0，`state_dict` keys 一致 |
| `use_mlka=True, base_num_features=30`，输入 `(1,3,64,64)`、`(2,3,96,128)`、`(1,3,128,96)` | 5 个深监督输出 shape 与 baseline 一致，15 个嵌套节点均为 `MLKABlock`，反向传播梯度正常 |
| 错误配置：`base_num_features=32`、`convolutional_upsampling=False`、`conv_op=nn.Conv3d` | 均按预期失败，错误原因明确 |

### 9. 改动一 Todo List

| 任务 | 状态 | 目标/验收 |
|------|:---:|----------|
| 修正 `generic_UNetPlusPlus.py` 中 `self.upsample_mode` 未赋值问题 | ✅ | 在 `__init__()` 中保存 `self.upsample_mode = upsample_mode`，避免非卷积上采样路径潜在报错 |
| 创建 `custom_modules/mlka.py` | ✅ | 提取并适配 GroupGLKA，默认 `InstanceNorm2d`，无 sigmoid gate |
| 为 `MLKABlock` 增加 `output_channels` | ✅ | `seg_outputs` 可正常构建，无 AttributeError |
| 在 `Generic_UNetPlusPlus.__init__()` 末尾追加 MLKA 参数 | ✅ | `use_mlka=False` 默认保持旧行为，位置参数调用不受影响 |
| 改造 `create_nest()` 条件分支 | ✅ | `use_mlka=True` 时第二步为 `MLKABlock(C_i)`；`False` 时保留原始 `final_num_features` 逻辑 |
| 创建 `nnUNetPlusPlusTrainerV2_MLKA` | ✅ | 独立 trainer 显式传入 `use_mlka=True, mlka_norm='instance'` |
| 随机张量前向/反向验证 | ✅ | `num_pool=5, base_num_features=30, 2D, conv_upsampling=True` 下输出 shape 与 baseline 一致；`base_num_features=32` 不属于当前改动一支持范围 |
| baseline 兼容性验证 | ✅ | `use_mlka=False` 在相同随机种子下与原始网络结构/输出一致；已对 `(1,3,64,64)` 和 `(2,3,96,128)` 验证最大差值为 0 |
| 参数量与显存记录 | ⬜ | 记录 baseline 与 MLKA 版本的参数量、显存峰值、单次 forward 时间 |

---

## 改动二：用上采样语义特征门控主 skip（GSAU-inspired）

**目标**：在每个 UNet++ 嵌套节点 concat 之前，用上采样后的深层语义特征生成空间 gate，门控该节点对应的主跳跃连接特征 `x^{i,0}`。

**命名说明**：这里的模块是面向 UNet++ skip 筛选任务设计的 **GSAU-inspired semantic skip gate**。它借鉴 MAN/SGAB 中“空间门控 + 逐元素调制”的思想，但为了让 skip 初始状态接近原始直通，当前方案采用 `sigmoid` gate 与正 bias 初始化；这与 MAN 原始 SGAB 中无 sigmoid 的乘法门控不是完全同一个实现。论文表述时应写成“受 GSAU 启发的语义门控单元”或明确说明是适配版 GSAU。

### 1. 集成原则

GSAU 仅作用于每个 UNet++ 节点输入中的**主跳跃连接** `x^{i,0}`。密集路径中的历史节点 `x^{i,1}, x^{i,2}, ...` 和上采样特征 `up_feat` 保持不变。

```
up_feat = Upsample(x_deep)
skip_gated = GSAU(gate_x=up_feat, target_y=x^{i,0})
node_input = concat([skip_gated, x^{i,1}, ..., x^{i,j-1}, up_feat])
x^{i,j} = self.loc*(node_input)
```

该方案不改变 `self.loc*[idx]` 的输入通道数，也不改变深监督头、`create_nest()`、`MLKABlock.output_channels` 的逻辑。

### 2. GSAU 模块接口

路径：`pytorch/nnunet/network_architecture/custom_modules/gsau.py`

建议实现：

```python
class GSAU(nn.Module):
    """
    gate_x:   [B, C, H, W]  上采样后的深层特征
    target_y: [B, C, H, W]  主跳跃连接 x^{i,0}
    output:   [B, C, H, W]
    """
    def __init__(self, channels, kernel_size=7, init_bias=2.0, weight_std=1e-3):
        ...

    def forward(self, gate_x, target_y):
        gate = torch.sigmoid(self.dwconv(gate_x))
        return target_y * gate
```

关键约束：
- `gate_x` 和 `target_y` 必须同 shape：`[B, C_i, H_i, W_i]`
- `GSAU(C_i)` 输出仍为 `[B, C_i, H_i, W_i]`
- 深度卷积 `groups=C_i`
- `dwconv.bias` 初始化为 `+2.0`，使 `sigmoid(2)≈0.88`，初始更接近保留跳跃连接
- `dwconv.weight` 使用很小的正态初始化（默认 `std=1e-3`），使 gate 初始接近常数但仍能从第一个 backward 向 `gate_x` 传递非零梯度
- 当前只实现 2D：`nn.Conv2d`
- 若后续想严格复现 MAN 的 SGAB/GSAU，可单独增加无 sigmoid 版本作为扩展消融，不混入当前主实验

### 3. forward 插入点与维度表

当前 `forward()` 固定 5-pool UNet++，共有 15 个嵌套节点。每个节点只替换 concat 中的第一个主跳跃连接项。

| 节点 | 原 concat 输入 | GSAU 后 concat 输入 | concat 通道 | loc 输出 |
|------|----------------|--------------------|------------|----------|
| `x0_1` | `[x0_0(30), up(x1_0)(30)]` | `[GSAU0(up(x1_0), x0_0)(30), up(x1_0)(30)]` | 60 | 30 |
| `x1_1` | `[x1_0(60), up(x2_0)(60)]` | `[GSAU1(up(x2_0), x1_0)(60), up(x2_0)(60)]` | 120 | 60 |
| `x0_2` | `[x0_0(30), x0_1(30), up(x1_1)(30)]` | `[GSAU0(up(x1_1), x0_0)(30), x0_1(30), up(x1_1)(30)]` | 90 | 30 |
| `x2_1` | `[x2_0(120), up(x3_0)(120)]` | `[GSAU2(up(x3_0), x2_0)(120), up(x3_0)(120)]` | 240 | 120 |
| `x1_2` | `[x1_0(60), x1_1(60), up(x2_1)(60)]` | `[GSAU1(up(x2_1), x1_0)(60), x1_1(60), up(x2_1)(60)]` | 180 | 60 |
| `x0_3` | `[x0_0(30), x0_1(30), x0_2(30), up(x1_2)(30)]` | `[GSAU0(up(x1_2), x0_0)(30), x0_1(30), x0_2(30), up(x1_2)(30)]` | 120 | 30 |
| `x3_1` | `[x3_0(240), up(x4_0)(240)]` | `[GSAU3(up(x4_0), x3_0)(240), up(x4_0)(240)]` | 480 | 240 |
| `x2_2` | `[x2_0(120), x2_1(120), up(x3_1)(120)]` | `[GSAU2(up(x3_1), x2_0)(120), x2_1(120), up(x3_1)(120)]` | 360 | 120 |
| `x1_3` | `[x1_0(60), x1_1(60), x1_2(60), up(x2_2)(60)]` | `[GSAU1(up(x2_2), x1_0)(60), x1_1(60), x1_2(60), up(x2_2)(60)]` | 240 | 60 |
| `x0_4` | `[x0_0(30), x0_1(30), x0_2(30), x0_3(30), up(x1_3)(30)]` | `[GSAU0(up(x1_3), x0_0)(30), x0_1(30), x0_2(30), x0_3(30), up(x1_3)(30)]` | 150 | 30 |
| `x4_1` | `[x4_0(480), up(x5_0)(480)]` | `[GSAU4(up(x5_0), x4_0)(480), up(x5_0)(480)]` | 960 | 480 |
| `x3_2` | `[x3_0(240), x3_1(240), up(x4_1)(240)]` | `[GSAU3(up(x4_1), x3_0)(240), x3_1(240), up(x4_1)(240)]` | 720 | 240 |
| `x2_3` | `[x2_0(120), x2_1(120), x2_2(120), up(x3_2)(120)]` | `[GSAU2(up(x3_2), x2_0)(120), x2_1(120), x2_2(120), up(x3_2)(120)]` | 480 | 120 |
| `x1_4` | `[x1_0(60), x1_1(60), x1_2(60), x1_3(60), up(x2_3)(60)]` | `[GSAU1(up(x2_3), x1_0)(60), x1_1(60), x1_2(60), x1_3(60), up(x2_3)(60)]` | 300 | 60 |
| `x0_5` | `[x0_0(30), x0_1(30), x0_2(30), x0_3(30), x0_4(30), up(x1_4)(30)]` | `[GSAU0(up(x1_4), x0_0)(30), x0_1(30), x0_2(30), x0_3(30), x0_4(30), up(x1_4)(30)]` | 180 | 30 |

通道结论：
- GSAU 不改变任何 concat 通道数
- `create_nest()` 中 `n_features_after_tu_and_concat` 不需要改
- `loc0..loc4`、`seg_outputs` 不需要因 GSAU 改通道
- 可复用 5 个尺度级 GSAU：`gsau_blocks[0..4]` 对应 `30,60,120,240,480`

### 4. 建议代码组织

在 `Generic_UNetPlusPlus.__init__()` 中追加参数，仍放在签名末尾，避免破坏旧的位置参数调用：

```python
use_gsau=False,
gsau_kernel_size=7,
gsau_init_bias=2.0,
gsau_weight_std=1e-3
```

建议增加轻量辅助函数，避免在 forward 中写大量重复条件。注意：不要写成 `self.apply_gsau(self.gsau0, ...)` 且只在 `use_gsau=True` 时注册 `self.gsau0`，因为 Python 会先访问 `self.gsau0`，导致 `use_gsau=False` 的 baseline 路径报 AttributeError。

```python
def apply_gsau(self, level, gate_x, target_y):
    if not self.use_gsau:
        return target_y
    return self.gsau_blocks[level](gate_x, target_y)
```

模块注册建议从 `self.conv_blocks_context[i].output_channels` 动态读取通道；当前主实验限定 `base_num_features=30`，对应通道为 `30,60,120,240,480`。`use_gsau=False` 时 `self.gsau_blocks = None`，保证 baseline 的 `state_dict` 不新增 GSAU 参数。

```python
self.gsau_blocks = None
if self.use_gsau:
    gsau_chs = [self.conv_blocks_context[i].output_channels for i in range(num_pool)]
    # base_num_features=30 时：gsau_chs == [30, 60, 120, 240, 480]
    self.gsau_blocks = nn.ModuleList([
        GSAU(
            ch,
            kernel_size=gsau_kernel_size,
            init_bias=gsau_init_bias,
            weight_std=gsau_weight_std,
        )
        for ch in gsau_chs
    ])
```

### 5. forward 代码变换模板

15 个节点手工改动时，先把上采样结果保存为局部变量，再把 concat 第一项替换为 `apply_gsau(...)` 的输出，避免下标误写。

**j=1，2 路 concat：**

```python
# before
x0_1 = self.loc4[0](torch.cat([x0_0, self.up4[0](x1_0)], 1))

# after
up_x1_0 = self.up4[0](x1_0)
x0_1 = self.loc4[0](torch.cat([self.apply_gsau(0, up_x1_0, x0_0), up_x1_0], 1))
```

**j=3，4 路 concat：**

```python
# before
x0_3 = self.loc2[2](torch.cat([x0_0, x0_1, x0_2, self.up2[2](x1_2)], 1))

# after
up_x1_2 = self.up2[2](x1_2)
x0_3 = self.loc2[2](torch.cat([
    self.apply_gsau(0, up_x1_2, x0_0), x0_1, x0_2, up_x1_2
], 1))
```

**j=5，6 路 concat：**

```python
# before
x0_5 = self.loc0[4](torch.cat([x0_0, x0_1, x0_2, x0_3, x0_4, self.up0[4](x1_4)], 1))

# after
up_x1_4 = self.up0[4](x1_4)
x0_5 = self.loc0[4](torch.cat([
    self.apply_gsau(0, up_x1_4, x0_0), x0_1, x0_2, x0_3, x0_4, up_x1_4
], 1))
```

### 6. 与 MLKA 的开关关系

| 配置 | 含义 |
|------|------|
| `use_mlka=False, use_gsau=False` | 原始 UNet++ baseline |
| `use_mlka=True, use_gsau=False` | 仅改动一 |
| `use_mlka=False, use_gsau=True` | 仅 GSAU |
| `use_mlka=True, use_gsau=True` | MLKA + GSAU |

GSAU 插在 `forward()` 的 concat 前；MLKA 位于 `create_nest()` 的节点精炼层。两者位置不同，可以独立开关。

### 7. 改动二 Todo List

| 任务 | 状态 | 目标/验收 |
|------|:---:|----------|
| 明确 GSAU 版本定义 | ✅ | 文档和代码注释均说明当前是 sigmoid-gated skip gate，属于 GSAU-inspired 适配版，不与 MAN 原始 SGAB 混淆 |
| 创建 `custom_modules/gsau.py` | ✅ | 已实现 2D `GSAU`，接口 `forward(gate_x, target_y)`，输出 shape 等于 `target_y` |
| 验证 `GSAU` 模块级维度与初始化 | ✅ | 已在 d2l 环境验证 5 个尺度：输出 shape 正确，gate 均值约 `0.880797≈sigmoid(2)`，`gate_x/target_y/weight` 梯度均正常 |
| 在 `Generic_UNetPlusPlus.__init__()` 末尾追加 GSAU 参数 | ✅ | 已追加 `use_gsau`、`gsau_kernel_size`、`gsau_init_bias`、`gsau_weight_std`，默认保持旧行为，位置参数调用不受影响 |
| 注册 5 个尺度级 GSAU 模块 | ✅ | 已使用 `self.gsau_blocks = nn.ModuleList([...])`，通道从 `conv_blocks_context` 动态读取；`use_gsau=False` 时为 `None` 且不新增参数 |
| 增加 `apply_gsau(level, gate_x, target_y)` 辅助函数 | ✅ | 已实现关闭时直接返回原 skip，开启时调用对应尺度 GSAU，且不会访问未注册的 GSAU 属性 |
| 修改 forward 的 15 个 concat 输入 | ✅ | 已保存每个上采样结果为局部变量，并仅替换 concat 的第一个 `x{i}_0` 为门控后的 skip；concat 通道数保持表中数值 |
| 创建仅 GSAU Trainer | ✅ | 已新增 `nnUNetPlusPlusTrainerV2_GSAU`，除 `use_mlka=False, use_gsau=True` 外与 baseline trainer 保持一致 |
| 创建 MLKA+GSAU Trainer | ✅ | 已新增 `nnUNetPlusPlusTrainerV2_MLKA_GSAU`，除 `use_mlka=True, use_gsau=True` 外与 baseline/MLKA trainer 保持一致 |
| 随机张量前向/反向验证 | ✅ | `base_num_features=30, num_pool=5, 2D, conv_upsampling=True` 下 5 个输出 shape 与 baseline 一致，梯度非 None |
| baseline 兼容性验证 | ✅ | `use_mlka=False,use_gsau=False` 与默认关闭路径在相同随机种子下输出最大差值为 0，`state_dict` 不新增 GSAU 参数 |
| 四组消融开关验证 | ✅ | Baseline、MLKA only、GSAU only、MLKA+GSAU 均可实例化、前向、反向；输出 shape 一致 |
| 主实验资源记录 | ⬜ | 四组均记录 Params、FLOPs/MACs、单次 forward 时间、显存峰值，避免只报告精度 |
| 错误配置验证 | ✅ | 3D 或 `convolutional_upsampling=False` 时给出明确限制；`base_num_features=32` 仅在 `use_mlka=True` 时不属于当前支持范围，GSAU-only 可单独验证 |

### 8. 改动二前三步模块级验证记录

验证环境：`D:\anaconda3\envs\d2l\python.exe`

| 通道 C | 输入尺寸 | 参数量 | 输出 shape | gate 范围 | gate 均值 | 梯度 |
|------:|----------|------:|------------|-----------|-----------|------|
| 30 | `[2,30,32,40]` | 1,500 | 正确 | `[0.8776, 0.8838]` | 0.880793 | `gate_x/target_y/weight` 均非零 |
| 60 | `[1,60,24,32]` | 3,000 | 正确 | `[0.8773, 0.8843]` | 0.880793 | `gate_x/target_y/weight` 均非零 |
| 120 | `[1,120,16,20]` | 6,000 | 正确 | `[0.8777, 0.8836]` | 0.880796 | `gate_x/target_y/weight` 均非零 |
| 240 | `[1,240,12,16]` | 12,000 | 正确 | `[0.8778, 0.8836]` | 0.880798 | `gate_x/target_y/weight` 均非零 |
| 480 | `[1,480,8,12]` | 24,000 | 正确 | `[0.8779, 0.8841]` | 0.880799 | `gate_x/target_y/weight` 均非零 |

5 个尺度级 GSAU 合计参数量：46,500。错误输入已验证：偶数 kernel、`gate_x/target_y` shape 不一致、通道不一致、负 `weight_std` 均会抛出明确 `ValueError`。

### 9. 改动二第 4-6 步网络构造验证记录

验证环境：`D:\anaconda3\envs\d2l\python.exe`，已安装 `batchgenerators`。

| 案例 | 结果 | 说明 |
|------|------|------|
| `use_gsau=False` | 通过 | `gsau_blocks=None`，`state_dict` 不含 `gsau_blocks`，`apply_gsau()` 直接返回原 skip |
| `use_gsau=True, base_num_features=30` | 通过 | 注册 5 个 GSAU，通道 `[30,60,120,240,480]`，参数量 46,500 |
| `use_gsau=True, base_num_features=32, use_mlka=False` | 通过 | GSAU-only 不继承 MLKA 的 3 分支整除限制，通道 `[32,64,128,256,480]`，参数量 48,000 |
| `use_gsau=True, use_mlka=True, base_num_features=32` | 按预期失败 | 抛出 `ValueError: channels must be divisible by num_groups for GroupGLKA.`，失败原因来自 MLKA 而非 GSAU |
| `apply_gsau(level=0)` | 通过 | 输入 `[2,30,32,40]`，输出 shape 正确，`gate_x/target_y` 梯度非零 |
| `apply_gsau(level=4)` | 通过 | 输入 `[2,480,4,5]`，输出 shape 正确，`gate_x/target_y` 梯度非零 |
| `convolutional_upsampling=False, use_gsau=True` | 按预期失败 | 抛出 `NotImplementedError: GSAU skip gating currently requires convolutional_upsampling=True.` |
| `conv_op=nn.Conv3d, use_gsau=True` | 按预期失败 | 抛出 `NotImplementedError: GSAU skip gating currently supports 2D Conv2d only.` |

### 10. 改动二第 7-8 步验证记录

验证环境：`D:\anaconda3\envs\d2l\python.exe`。为兼容当前 `batchgenerators` 版本，已对 data augmentation / dataloading 相关导入增加 fallback，不改变训练逻辑。

| 案例 | 结果 | 说明 |
|------|------|------|
| `use_gsau=False, base_num_features=30` 前向/反向 | 通过 | 5 个深监督输出 shape 均为 `[1,2,64,64]`，`gsau_blocks=None` |
| `use_gsau=True, base_num_features=30` 前向/反向 | 通过 | 5 个深监督输出 shape 均为 `[1,2,64,64]`，5 个 GSAU 均注册，GSAU 参数梯度非零 |
| `use_gsau=True, base_num_features=32, use_mlka=False` 前向/反向 | 通过 | 5 个深监督输出 shape 均为 `[1,2,64,64]`，说明 GSAU-only 不依赖 MLKA 的 3 分支通道约束 |
| `nnUNetPlusPlusTrainerV2_GSAU.initialize_network()` | 通过 | 初始化网络得到 `use_mlka=False, use_gsau=True`，GSAU 通道 `[30,60,120,240,480]`，参数量 46,500 |

### 11. 改动二第 9 步验证记录

验证环境：`D:\anaconda3\envs\d2l\python.exe`。

| 案例 | 结果 | 说明 |
|------|------|------|
| `use_mlka=True, use_gsau=True, base_num_features=30` 前向/反向 | 通过 | 5 个深监督输出 shape 均为 `[1,2,64,64]`，GSAU 参数梯度非零，MLKA `scale` 梯度非零 |
| `nnUNetPlusPlusTrainerV2_MLKA_GSAU.initialize_network()` | 通过 | 初始化网络得到 `use_mlka=True, use_gsau=True`，GSAU 通道 `[30,60,120,240,480]` |

### 12. 四组消融级随机张量验证记录

验证环境：`D:\anaconda3\envs\d2l\python.exe`，输入 `[1,3,64,64]`，`base_num_features=30`。

| 实验组 | 输出 shape | 参数量 | GSAU 参数 | 状态与梯度 |
|------|------------|------:|----------:|------------|
| Baseline | 5 个 `[1,2,64,64]` | 22,836,630 | 0 | 无 MLKA/GSAU state |
| MLKA only | 5 个 `[1,2,64,64]` | 20,721,270 | 0 | MLKA state 存在，MLKA `scale` 梯度非零 |
| GSAU only | 5 个 `[1,2,64,64]` | 22,883,130 | 46,500 | GSAU state 存在，GSAU 梯度非零 |
| MLKA + GSAU | 5 个 `[1,2,64,64]` | 20,767,770 | 46,500 | MLKA/GSAU state 均存在，梯度非零 |

额外验证：

| 案例 | 结果 |
|------|------|
| 默认参数 vs `use_mlka=False,use_gsau=False` | 相同随机种子下输出最大差值 0 |
| `GSAU only, base_num_features=32` | 通过，参数量 24,048,608，GSAU 参数 48,000 |
| `MLKA only, base_num_features=32` | 按预期失败：`channels must be divisible by num_groups for GroupGLKA.` |
| `MLKA + GSAU, base_num_features=32` | 按预期失败：`channels must be divisible by num_groups for GroupGLKA.` |
| `nnUNetPlusPlusTrainerV2_MLKA` | 初始化得到 `(use_mlka=True, use_gsau=False)` |
| `nnUNetPlusPlusTrainerV2_GSAU` | 初始化得到 `(use_mlka=False, use_gsau=True)` |
| `nnUNetPlusPlusTrainerV2_MLKA_GSAU` | 初始化得到 `(use_mlka=True, use_gsau=True)` |

### 13. 改动二限制

| 限制 | 说明 |
|------|------|
| 仅 2D | 当前 GSAU 使用 `nn.Conv2d` |
| 依赖上采样后通道与主 skip 通道一致 | 当前 `convolutional_upsampling=True` 时成立 |
| 仍继承改动一的主实验边界 | `num_pool=5`、`base_num_features=30`、2D |
| 不改变 dense skip | 只门控 `x^{i,0}`，不门控 `x^{i,1}...x^{i,j-1}` |

---

## 代码质量标准（强制约束）

### 封装性
- 所有新模块放在 `pytorch/nnunet/network_architecture/custom_modules/` 目录下，每个模块一个文件
- 模块接口清晰：输入 → 模块 → 输出，不依赖外部全局状态
- `generic_UNetPlusPlus.py` 仅通过 `import` 使用新模块，不内联复杂逻辑

### 可读性
- 类名使用 PascalCase，函数/变量名使用 snake_case
- 每个新增类必须有单行 docstring 说明其用途
- 关键参数（如 `kernel_configs`、`num_groups`）使用具名常量或配置字典，避免魔法数字
- 前向传播中的 GSAU/MLKA 插入点用明确的条件分支，且分支内代码不超过 5 行

### 代码优雅
- 优先复用 `MAN_arch.py` 中已验证的模块代码，不重新发明轮子
- 所有新增模块支持 `nn.ModuleList` 注册（确保 `model.parameters()` 和 `model.cuda()` 正常工作）
- 避免深层嵌套（最多 3 层 `nn.Sequential`）
- 使用 `copy.deepcopy` 处理可变默认参数（与现有 nnU-Net 代码风格一致）
- `MLKABlock` 与 `GroupGLKA` 的关系是**适配**而非包装：在 `custom_modules/mlka.py` 中复制并适配 GroupGLKA，不额外嵌套一层

### 兼容性
- **所有改动必须保持向后兼容**：通过 feature flag（`use_mlka`, `use_gsau`）控制，默认 `False` 时行为与原实现完全一致
- 不修改 nnU-Net 训练管线（`run_training.py`）
- 不修改 `MAN_arch.py`
- 不破坏深度监督机制（5 个分割头保留，`seg_outputs` 列表结构不变）

### 测试要求
- 每次改动后运行：实例化网络 → 随机输入前向传播 → 反向传播 → 检查梯度非 None
- 当前支持范围与已验证案例见“改动一限制与实验边界”章节

---

## 进度追踪

| 任务 | 状态 | 开始日期 | 完成日期 | 备注 |
|------|:----:|----------|----------|------|
| 改动一：创建 custom_modules/mlka.py | ✅ 已完成 | 2026-05-25 | 2026-05-25 | 可配置 Norm + GroupGLKA + MLKABlock，已包含 output_channels |
| 改动一：generic_UNetPlusPlus.py 导入与参数 | ✅ 已完成 | 2026-05-25 | 2026-05-25 | 已追加 use_mlka, mlka_groups, mlka_norm，并加入 MLKABlock 导入 |
| 改动一：create_nest 条件分支改造 | ✅ 已完成 | 2026-05-25 | 2026-05-25 | `use_mlka=True` 时第二步 C_i→C_i 使用 MLKABlock |
| 改动一：创建 nnUNetPlusPlusTrainerV2_MLKA | ✅ 已完成 | 2026-05-25 | 2026-05-25 | 独立 trainer，传入 use_mlka=True |
| 改动一：前后向兼容性验证 | ✅ 已完成 | 2026-05-25 | 2026-05-25 | 已验证 baseline 数值一致、MLKA 多尺寸前后向、错误配置保护 |
| 改动二：创建 custom_modules/gsau.py | ✅ 已完成 | 2026-05-27 | 2026-05-27 | GSAU-inspired sigmoid skip gate，模块级验证通过 |
| 改动二：forward 方法 GSAU 插入 | ✅ 已完成 | 2026-05-27 | 2026-05-27 | 方案 A，仅作用于 x^{i,0}；前向/反向验证通过 |
| 改动二：创建 nnUNetPlusPlusTrainerV2_GSAU | ✅ 已完成 | 2026-05-27 | 2026-05-27 | 仅 GSAU trainer，`use_mlka=False,use_gsau=True` |
| 改动二：创建 nnUNetPlusPlusTrainerV2_MLKA_GSAU | ✅ 已完成 | 2026-05-27 | 2026-05-27 | 联合 trainer，`use_mlka=True,use_gsau=True` |
| 改动二：独立开关验证 | ✅ 已完成 | 2026-05-27 | 2026-05-27 | 四组消融开关均可前向/反向 |
| MLKA + GSAU 联合测试 | ✅ 已完成 | 2026-05-27 | 2026-05-27 | 联合模型前向/反向和 trainer 初始化通过 |
| 训练管线集成测试 | ⬜ 待开始 | — | — | |

状态说明：⬜ 待开始 | 🔄 进行中 | ✅ 已完成 | ❌ 已阻塞

---

## 注意事项

1. **先改一后改二**：改动二是改动一的增强，依赖改动一完成后的 MLKA 接口与 GSAU 协同。不要并行开发。
2. **每次改动后更新本文档**：在对应任务行更新状态、日期和备注。
3. **归一化层选择**：当前决策为 MLKA 内部默认使用 InstanceNorm2d（与 `nnUNetPlusPlusTrainerV2` 一致）。可通过 `mlka_norm` 参数切换为 `'layer'`（MAN 原风格）或 `'batch'` 做消融实验。
4. **不修改 MAN_arch.py**：该文件作为参考实现保留不动。所需模块提取到 `custom_modules/` 后独立维护。
5. **nnU-Net 的 `deep_supervision` 参数**：改动不得影响 `_deep_supervision` 和 `do_ds` 的逻辑流程。
6. **MLKABlock 必须有 `output_channels` 属性**：`Generic_UNetPlusPlus.__init__()` 在构建 `seg_outputs` 时依赖 `self.loc*[-1][-1].output_channels`（L348-L357），缺失会导致 AttributeError。
7. **实验边界集中维护**：2D、`convolutional_upsampling=True`、`base_num_features=30`、`num_pool=5` 等限制统一以“改动一限制与实验边界”章节为准。

---

## 实验设计

### 1. 实验目的

验证在 UNet++ 中引入 MLKA 与 GSAU 后，模型性能提升是否来自模块本身，而不是随机种子、训练轮数、数据划分或参数量变化带来的偶然波动。

核心问题：

1. MLKA 替换节点内部第二个 3×3 精炼卷积后，是否提升多尺度上下文建模能力？
2. GSAU 对原始 skip feature 做语义引导门控后，是否减少浅层噪声注入？
3. MLKA 与 GSAU 同时使用时，是互补提升、无明显叠加，还是互相干扰？

### 2. 主消融实验

保持数据集、预处理、patch size、batch size、优化器、学习率策略、训练 epoch、数据增强、fold 划分和 deep supervision 设置完全一致，仅改变 `use_mlka` 与 `use_gsau` 两个开关。

| 实验组 | `use_mlka` | `use_gsau` | 目的 |
|------|:----------:|:----------:|------|
| Baseline | False | False | 原始 UNet++ 对照组 |
| MLKA only | True | False | 验证节点内部多尺度大核精炼的独立贡献 |
| GSAU only | False | True | 验证 skip 进入 concat 前语义门控的独立贡献 |
| MLKA + GSAU | True | True | 验证二者组合是否互补 |

当前代码进度下，Baseline 与 MLKA only 可优先执行；GSAU only 和 MLKA + GSAU 需要等改动二实现并通过前后向验证后再执行。

### 3. 训练设置

推荐主实验固定如下边界：

| 项目 | 设置 |
|------|------|
| 任务类型 | 2D 语义分割 |
| 网络 | `Generic_UNetPlusPlus` |
| `base_num_features` | 30 |
| `num_pool` | 5 |
| 上采样 | `convolutional_upsampling=True` |
| 深度监督 | 保持原 UNet++ 设置 |
| MLKA 归一化 | `mlka_norm='instance'` |
| MLKA 分支数 | `mlka_groups=3` |
| 随机种子 | 至少 3 个 seed，推荐 5 个 |

所有实验组应使用相同的数据划分。若使用 nnU-Net 的 cross-validation，应至少报告每个 fold 的结果和平均结果；若只跑单 fold，应明确标注 fold 编号，并避免把单次结果写成稳定结论。

### 4. 评价指标

主指标：

| 指标 | 说明 |
|------|------|
| Dice | 主分割精度指标 |
| IoU / mIoU | 区域重叠质量 |
| HD95 | 边界最坏情况稳定性 |
| ASD / ASSD | 平均边界距离 |

辅助指标：

| 指标 | 说明 |
|------|------|
| 参数量 | 比较结构复杂度 |
| FLOPs / MACs | 比较计算量 |
| 单张推理时间 | 比较实际速度 |
| 显存峰值 | 比较训练和推理资源消耗 |
| 收敛曲线 | 比较训练稳定性和过拟合情况 |

如果数据集中存在小目标、细长目标或大面积目标，建议额外按目标尺度分组统计 Dice / IoU，以判断 MLKA 的多尺度建模是否真正带来收益。

### 5. 结果记录模板

| 实验组 | Seed/Fold | Dice | IoU | HD95 | ASD | Params | FLOPs | Inference Time | Peak Memory |
|------|-----------|------|-----|------|-----|--------|-------|----------------|-------------|
| Baseline | | | | | | | | | |
| MLKA only | | | | | | | | | |
| GSAU only | | | | | | | | | |
| MLKA + GSAU | | | | | | | | | |

最终汇总时报告 `mean ± std`。如果 MLKA 或 GSAU 只在某个 seed 上提升，而均值和方差不稳定，不能直接得出“模块有效”的结论。

### 6. 结论判定标准

可以认为模块有效的条件：

1. 验证集或测试集主指标稳定提升，而不是只提升训练集指标。
2. 多个 seed 或多个 fold 的均值提升，且方差可接受。
3. 提升幅度大于随机波动，并尽量通过统计检验或置信区间说明。
4. 额外参数量、FLOPs、显存和推理时间的增加在可接受范围内。
5. 可视化结果中能看到合理改善，例如小目标漏分减少、边界更连续、背景误检降低。

不能认为模块有效的情况：

1. 训练集提升但验证集不提升，可能是过拟合。
2. Dice 提升但 HD95 / ASD 明显变差，说明边界质量可能受损。
3. 只有 MLKA + GSAU 提升，但 MLKA only 和 GSAU only 都不提升，需要进一步分析是否是训练波动。
4. 指标提升很小，但计算量和推理时间显著增加，需要重新评估性价比。

### 7. 预期现象与分析方向

| 现象 | 可能解释 |
|------|----------|
| MLKA only 提升，小目标和大结构均改善 | 多尺度大核精炼有效增强上下文表达 |
| MLKA only 不提升或边界变差 | 大感受野可能平滑细节，或当前任务更依赖局部纹理 |
| GSAU only 提升，误检减少 | 深层语义门控减少了浅层 skip 噪声 |
| GSAU only 小目标下降 | gate 可能压制了弱目标或细边界 |
| MLKA + GSAU 高于两者单独结果 | 后融合精炼与前融合筛选存在互补 |
| MLKA + GSAU 低于单模块 | 两种门控/注意力机制可能过强，需要调 gate 初始化、学习率或正则化 |

### 8. 论文写作建议

实验结论应使用谨慎表述：

- 可以写：“MLKA 在本数据集上带来了稳定提升，说明多尺度大核空间建模对该任务有效。”
- 可以写：“GSAU 减少了浅层 skip 中的无关纹理干扰，在误检较多的类别上改善明显。”
- 不建议写：“MLKA/GSAU 一定优于原始 UNet++。”
- 不建议只用单次实验结果证明模块普适有效。
