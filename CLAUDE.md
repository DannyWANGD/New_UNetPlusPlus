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

### 8. 已知限制与后续扩展

| 限制 | 说明 | 扩展方案 |
|------|------|----------|
| **仅支持 2D** | `MAN_arch.py` 全系使用 `nn.Conv2d`，无法直接用于 3D fullres | 需实现 Conv3d 版 MLKA：Conv2d→Conv3d, InstanceNorm2d→InstanceNorm3d, kernel 适配 3D（如 (3,7,7)） |
| **依赖 conv_upsampling=True** | `final_num_features = nfeatures_from_skip` 保证第二步为 C_i→C_i | 若需支持 conv_upsampling=False，修改 MLKABlock 签名支持 `MLKABlock(in_ch, out_ch)`，末尾加 1×1 projection |
| **需独立 Trainer** | `use_mlka` 默认为 False，原 trainer 不传此参数 | 需创建 `nnUNetPlusPlusTrainerV2_MLKA` 并显式传入 `use_mlka=True` |

### 9. 改动一 Todo List

| 任务 | 状态 | 目标/验收 |
|------|:---:|----------|
| 修正 `generic_UNetPlusPlus.py` 中 `self.upsample_mode` 未赋值问题 | ⬜ | 在 `__init__()` 中保存 `self.upsample_mode = upsample_mode`，避免非卷积上采样路径潜在报错 |
| 创建 `custom_modules/mlka.py` | ⬜ | 提取并适配 GroupGLKA，默认 `InstanceNorm2d`，无 sigmoid gate |
| 为 `MLKABlock` 增加 `output_channels` | ⬜ | `seg_outputs` 可正常构建，无 AttributeError |
| 在 `Generic_UNetPlusPlus.__init__()` 末尾追加 MLKA 参数 | ⬜ | `use_mlka=False` 默认保持旧行为，位置参数调用不受影响 |
| 改造 `create_nest()` 条件分支 | ⬜ | `use_mlka=True` 时第二步为 `MLKABlock(C_i)`；`False` 时保留原始 `final_num_features` 逻辑 |
| 创建 `nnUNetPlusPlusTrainerV2_MLKA` | ⬜ | 独立 trainer 显式传入 `use_mlka=True, mlka_norm='instance'` |
| 随机张量前向/反向验证 | ⬜ | `num_pool=5, base_num_features=30, 2D, conv_upsampling=True` 下输出 shape 与 baseline 一致 |
| baseline 兼容性验证 | ⬜ | `use_mlka=False` 在相同随机种子下与原始网络结构/输出一致 |
| 参数量与显存记录 | ⬜ | 记录 baseline 与 MLKA 版本的参数量、显存峰值、单次 forward 时间 |

---

## 改动二：解码器上采样后添加 GSAU

**目标**：在每个解码器上采样操作后，插入门控空间注意力单元，用上采样特征的空间信息门控主跳跃连接特征（x^{i,0}）。

**集成方案（方案 A — 推荐）**：
```
up_feat = Upsample(x_deep)
gsau_feat = GSAU(gate=up_feat, target=x_encoder)   # x_encoder = x^{i,0}
node_input = concat([gsau_feat, x^{i,1}, ..., x^{i,j-1}, up_feat])
x_out = MLKA(node_input)
```
GSAU 仅作用于主跳跃连接（编码器特征 x^{i,0}），密集路径中的其他节点（x^{i,1}, x^{i,2}...）保持不变。

**实施步骤**：

1. [ ] 在 `custom_modules/gsau.py` 中实现 `GSAU` 类：
   - 接口：`forward(gate_x, target_y)` → `sigmoid(DWConv(gate_x)) * target_y`
   - 深度卷积 bias 初始化为 +2.0（使 sigmoid 初始接近 1，即恒等映射）

2. [ ] （可选）同时实现 `SGAB`（`MAN_arch.py:233`）作为 GSAU 的增强替代：带 1×1 通道扩展 + SimpleGate + 残差

3. [ ] 在 `generic_UNetPlusPlus.__init__()` 中增加 `use_gsau: bool = False` 参数

4. [ ] 修改 `forward()` 方法：在每个上采样操作后、concat 之前插入 GSAU，仅作用于 x^{i,0}（编码器特征的跳跃连接）

5. [ ] 确保 GSAU 与 MLKA 可独立开关（支持消融实验：仅 MLKA / 仅 GSAU / 两者叠加）

**验收标准**：
- `use_gsau=False` 时行为不变
- `use_gsau=True` 时空间门控输出在 [0, 1] 范围内
- 参数量增幅 < 3%（仅 DWConv 参数）

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
- 当前 `forward()` 固定为 5-pool UNet++ 拓扑，因此先仅验证 `num_pool=5`
- 验证 `base_num_features` 选择可被 3 整除且每组不少于 8 通道的配置，如 24、30、48

---

## 进度追踪

| 任务 | 状态 | 开始日期 | 完成日期 | 备注 |
|------|:----:|----------|----------|------|
| 改动一：创建 custom_modules/mlka.py | ⬜ 待开始 | — | — | 可配置 Norm + GroupGLKA + MLKABlock |
| 改动一：generic_UNetPlusPlus.py 导入与参数 | ⬜ 待开始 | — | — | use_mlka, mlka_groups, mlka_norm='instance' |
| 改动一：create_nest 条件分支改造 | ⬜ 待开始 | — | — | L466-473，第二步 C_i→C_i |
| 改动一：创建 nnUNetPlusPlusTrainerV2_MLKA | ⬜ 待开始 | — | — | 独立 trainer，传入 use_mlka=True |
| 改动一：前后向兼容性验证 | ⬜ 待开始 | — | — | use_mlka=False 输出一致 + output_channels 正常 |
| 改动二：创建 custom_modules/gsau.py | ⬜ 待开始 | — | — | GSAU + SGAB |
| 改动二：forward 方法 GSAU 插入 | ⬜ 待开始 | — | — | 方案 A，仅作用于 x^{i,0} |
| 改动二：独立开关验证 | ⬜ 待开始 | — | — | |
| MLKA + GSAU 联合测试 | ⬜ 待开始 | — | — | |
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
7. **三分支通道限制**：当前 MLKA 复用 MAN GroupGLKA 的三分支结构，要求 `C_i % 3 == 0` 且每组不少于 8 通道。默认 `30, 60, 120, 240, 480` 满足要求；`base_num_features=32` 暂不支持。
8. **2D 限定**：当前所有 MLKA 模块基于 `nn.Conv2d`，不可直接用于 3D 网络。
