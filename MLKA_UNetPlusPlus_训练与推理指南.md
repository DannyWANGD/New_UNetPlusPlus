# MLKA UNet++ 训练与推理指南

本文档说明如何在 GPU 服务器上使用当前 UNet++ 改动版本，包括 `nnUNetPlusPlusTrainerV2_MLKA`、`nnUNetPlusPlusTrainerV2_GSAU` 与 `nnUNetPlusPlusTrainerV2_MLKA_GSAU`。

## 1. 当前支持范围

| 项目 | 要求 |
|------|------|
| 网络 | 仅 `2d` |
| Trainer | `nnUNetPlusPlusTrainerV2_MLKA` / `nnUNetPlusPlusTrainerV2_GSAU` / `nnUNetPlusPlusTrainerV2_MLKA_GSAU` |
| plans | 必须使用 `base_num_features=30` 的 2D plans |
| 上采样 | 必须 `convolutional_upsampling=True` |
| 拓扑 | 当前 `Generic_UNetPlusPlus.forward()` 固定 5-pool |
| 不支持 | 3D、`base_num_features=32`、非卷积上采样 |

不要直接使用默认 `nnUNetPlansv2.1` 的 2D plans。该 planner 默认 `base_num_features=32`，当前 MLKA 会报错。

## 2. 环境准备

进入仓库根目录：

```bash
cd /path/to/New_UNetPlusPlus
```

### 2.1 一行命令创建 conda 环境并安装依赖

推荐在仓库根目录下执行。下面命令会从新建 conda 环境开始，安装 PyTorch、项目依赖和当前仓库：

```bash
conda create -n unetpp_mlka python=3.9 -y && conda run -n unetpp_mlka python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118 && conda run -n unetpp_mlka python -m pip install -r requirements-unetpp.txt && conda run -n unetpp_mlka python -m pip install --no-deps -e ./pytorch
```

如果服务器 CUDA 版本不是 11.8，请把 `cu118` 替换为与你的服务器匹配的 PyTorch CUDA wheel，例如 `cu121`。如果只做 CPU 本地结构验证，可以去掉 `--index-url https://download.pytorch.org/whl/cu118`。

`requirements-unetpp.txt` 不包含 `dicom2nifti`。原因是新版 `dicom2nifti` 在 Windows 上会拉取 `python-gdcm`，可能触发本地 CMake/GDCM 编译失败。若你需要把 DICOM 转为 NIfTI，可在环境建好后额外安装：

```bash
conda run -n unetpp_mlka python -m pip install dicom2nifti
```

如果你的数据已经是 nnU-Net 标准 NIfTI 格式，不需要安装它。

安装本仓库时使用 `--no-deps -e ./pytorch`，是为了避免 `pytorch/setup.py` 中原始 nnU-Net 依赖再次拉取未固定版本的 `medpy` 或可选的 `dicom2nifti`。依赖以仓库根目录的 `requirements-unetpp.txt` 为准。

Windows PowerShell 中也可以使用同样的一行命令，只是通常用分号分隔：

```powershell
conda create -n unetpp_mlka python=3.9 -y; conda run -n unetpp_mlka python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118; conda run -n unetpp_mlka python -m pip install -r requirements-unetpp.txt; conda run -n unetpp_mlka python -m pip install --no-deps -e ./pytorch
```

### 2.2 分步安装方式

如果需要逐步排查环境问题，也可以分步执行：

```bash
conda create -n unetpp_mlka python=3.9 -y
conda activate unetpp_mlka
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements-unetpp.txt
pip install --no-deps -e ./pytorch
```

确认 PyTorch 可使用 GPU：

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## 3. nnU-Net 路径变量

设置三个目录：

```bash
export nnUNet_raw_data_base=/data/nnUNet_raw_data_base
export nnUNet_preprocessed=/data/nnUNet_preprocessed
export RESULTS_FOLDER=/data/nnUNet_results
export PYTHONPATH=/path/to/New_UNetPlusPlus/pytorch:$PYTHONPATH
```

如果已经通过 `pip install -e ./pytorch` 安装成功，通常不需要额外设置 `PYTHONPATH`；保留该变量主要用于确保命令行能加载当前仓库里的新增 trainer。

目录含义：

| 变量 | 用途 |
|------|------|
| `nnUNet_raw_data_base` | 原始数据根目录 |
| `nnUNet_preprocessed` | 预处理结果和 plans |
| `RESULTS_FOLDER` | 训练权重、日志、验证结果 |

## 4. 数据集放置格式

数据集必须放在：

```text
$nnUNet_raw_data_base/nnUNet_raw_data/TaskXXX_TaskName/
```

典型结构：

```text
TaskXXX_TaskName/
├── dataset.json
├── imagesTr/
│   ├── case001_0000.nii.gz
│   ├── case002_0000.nii.gz
│   └── ...
├── labelsTr/
│   ├── case001.nii.gz
│   ├── case002.nii.gz
│   └── ...
└── imagesTs/
    ├── case101_0000.nii.gz
    └── ...
```

命名规则：

| 文件 | 规则 |
|------|------|
| 训练图像 | `caseID_0000.nii.gz`，多模态则继续 `caseID_0001.nii.gz` |
| 训练标签 | `caseID.nii.gz` |
| 测试图像 | 与训练图像同格式 |
| `dataset.json` | 必须包含 modality、labels、numTraining、numTest 等字段 |

遥感多通道数据如果已经转换为 nnU-Net 接受的 NIfTI 多模态格式，每个通道按 `_0000/_0001/...` 放置。

## 5. 规划与预处理

当前 MLKA 必须使用 `base_num_features=30`。因此规划时使用旧版 2D planner。

`TASK_ID` 是任务编号，例如 `101`；`TaskXXX_TaskName` 是任务文件夹名，例如 `Task101_RoadSeg`。

```bash
nnUNet_plan_and_preprocess -t TASK_ID -pl2d ExperimentPlanner2D -pl3d None --verify_dataset_integrity
```

示例：

```bash
nnUNet_plan_and_preprocess -t 101 -pl2d ExperimentPlanner2D -pl3d None --verify_dataset_integrity
```

该命令会生成 2D plans：

```text
$nnUNet_preprocessed/TaskXXX_TaskName/nnUNetPlans_plans_2D.pkl
```

检查 plans 是否符合要求，注意替换实际任务名：

```bash
python -c "from batchgenerators.utilities.file_and_folder_operations import load_pickle; p=load_pickle('$nnUNet_preprocessed/TaskXXX_TaskName/nnUNetPlans_plans_2D.pkl'); print(p['base_num_features'])"
```

必须输出：

```text
30
```

如果输出 `32`，不要训练 MLKA；需要用 `ExperimentPlanner2D` 重新生成 2D plans。这里的 plans 是 nnU-Net 的网络规划文件，记录 patch size、pooling kernel、conv kernel、batch size、`base_num_features` 等配置，不是模型权重，也不是原始数据。

重新生成 plans 的含义是：保留同一份 `TaskXXX_TaskName` 数据集，重新运行规划/预处理命令，让 `$nnUNet_preprocessed/TaskXXX_TaskName/` 下生成 `nnUNetPlans_plans_2D.pkl`，其 `base_num_features=30`。

## 6. 启动训练

### 6.1 Baseline 训练（可选对照）

Baseline、MLKA 和 GSAU 是独立训练任务。Baseline 只用于做消融对照，不是运行改进模型的前置条件。不同 trainer 会写入不同结果目录。

如需对比原始 UNet++，运行：

```bash
CUDA_VISIBLE_DEVICES=0 nnUNet_train 2d nnUNetPlusPlusTrainerV2 TaskXXX_TaskName 0 -p nnUNetPlans
```

### 6.2 MLKA 训练

只想训练改进模型时，可以直接运行 MLKA：

```bash
CUDA_VISIBLE_DEVICES=0 nnUNet_train 2d nnUNetPlusPlusTrainerV2_MLKA TaskXXX_TaskName 0 -p nnUNetPlans
```

参数说明：

| 参数 | 含义 |
|------|------|
| `2d` | 使用 2D 网络 |
| `nnUNetPlusPlusTrainerV2_MLKA` | 当前改动一的独立 trainer |
| `TaskXXX_TaskName` | 任务名，也可用任务 ID，例如 `101` |
| `0` | 第 0 折，可改为 `1..4` 或 `all` |
| `-p nnUNetPlans` | 使用 `ExperimentPlanner2D` 生成的 30 通道 plans |

继续训练：

```bash
CUDA_VISIBLE_DEVICES=0 nnUNet_train 2d nnUNetPlusPlusTrainerV2_MLKA TaskXXX_TaskName 0 -p nnUNetPlans -c
```

只做验证：

```bash
CUDA_VISIBLE_DEVICES=0 nnUNet_train 2d nnUNetPlusPlusTrainerV2_MLKA TaskXXX_TaskName 0 -p nnUNetPlans -val
```

### 6.3 GSAU 训练

仅训练 GSAU skip gate 消融组：

```bash
CUDA_VISIBLE_DEVICES=0 nnUNet_train 2d nnUNetPlusPlusTrainerV2_GSAU TaskXXX_TaskName 0 -p nnUNetPlans
```

GSAU-only 本身不要求 `base_num_features` 必须能被 3 整除；但为了与 MLKA 和 Baseline 做公平消融，主实验仍建议统一使用 `ExperimentPlanner2D` 生成的 `base_num_features=30` plans。

### 6.4 MLKA + GSAU 联合训练

训练联合改进组：

```bash
CUDA_VISIBLE_DEVICES=0 nnUNet_train 2d nnUNetPlusPlusTrainerV2_MLKA_GSAU TaskXXX_TaskName 0 -p nnUNetPlans
```

联合组包含 MLKA，因此仍必须使用 `base_num_features=30` 的 2D plans。

## 7. 推理

推理输入目录必须只放待预测图像，命名仍为：

```text
caseID_0000.nii.gz
caseID_0001.nii.gz
...
```

单折推理：

```bash
CUDA_VISIBLE_DEVICES=0 nnUNet_predict \
  -i /data/inference/imagesTs \
  -o /data/inference/pred_mlka \
  -t TaskXXX_TaskName \
  -m 2d \
  -tr nnUNetPlusPlusTrainerV2_MLKA \
  -p nnUNetPlans \
  -f 0
```

`-tr`、`-m`、`-p` 必须和训练时一致，否则会找不到模型目录或加载错误配置。

如果推理 GSAU-only 模型，把 `-tr nnUNetPlusPlusTrainerV2_MLKA` 替换为：

```bash
-tr nnUNetPlusPlusTrainerV2_GSAU
```

如果推理 MLKA+GSAU 联合模型，替换为：

```bash
-tr nnUNetPlusPlusTrainerV2_MLKA_GSAU
```

自动使用已训练 folds：

```bash
CUDA_VISIBLE_DEVICES=0 nnUNet_predict \
  -i /data/inference/imagesTs \
  -o /data/inference/pred_mlka \
  -t TaskXXX_TaskName \
  -m 2d \
  -tr nnUNetPlusPlusTrainerV2_MLKA \
  -p nnUNetPlans
```

禁用 TTA 加速推理：

```bash
CUDA_VISIBLE_DEVICES=0 nnUNet_predict \
  -i /data/inference/imagesTs \
  -o /data/inference/pred_mlka_fast \
  -t TaskXXX_TaskName \
  -m 2d \
  -tr nnUNetPlusPlusTrainerV2_MLKA \
  -p nnUNetPlans \
  --disable_tta
```

## 8. 输出位置

训练输出：

```text
$RESULTS_FOLDER/nnUNet/2d/TaskXXX_TaskName/nnUNetPlusPlusTrainerV2_MLKA__nnUNetPlans/
```

常见文件：

```text
fold_0/
├── model_final_checkpoint.model
├── model_final_checkpoint.model.pkl
├── model_best.model
├── progress.png
└── training_log_*.txt
```

推理输出：

```text
/data/inference/pred_mlka/
├── case101.nii.gz
├── case102.nii.gz
└── ...
```

## 9. 限制与注意事项

1. 只能训练 `2d`，不要用 `3d_fullres`、`3d_lowres`、`3d_cascade_fullres`。
2. 必须使用 `-p nnUNetPlans`，前提是该 plans 由 `ExperimentPlanner2D` 生成，且 `base_num_features=30`。
3. 不要用默认 `nnUNetPlansv2.1` 训练 MLKA，2D v2.1 默认 `base_num_features=32`，当前不支持。
4. 当前 MLKA 固定 3 分支，要求节点通道能被 3 整除且每组不少于 8 通道。
5. 当前支持的主实验配置是 `base_num_features=30`，对应通道 `30, 60, 120, 240, 480`。
6. 推理时的 `-tr` 和 `-p` 必须与训练时一致。
7. 原始 baseline 使用 `nnUNetPlusPlusTrainerV2`；MLKA 使用 `nnUNetPlusPlusTrainerV2_MLKA`，不要混用结果目录。
8. 新增 trainer 文件 `pytorch/nnunet/training/network_training/nnUNetPlusPlusTrainerV2_MLKA.py` 必须同步到 GPU 服务器。

## 10. 常见错误

### `ValueError: channels must be divisible by num_groups`

原因：plans 中 `base_num_features=32`。

处理：

```bash
nnUNet_plan_and_preprocess -t TASK_ID -pl2d ExperimentPlanner2D -pl3d None
```

并训练时使用：

```bash
-p nnUNetPlans
```

### `Could not find trainer class`

原因：`nnUNetPlusPlusTrainerV2_MLKA.py` 没有同步到 GPU 服务器，或 `PYTHONPATH` 没指向当前仓库。

处理：

```bash
export PYTHONPATH=/path/to/New_UNetPlusPlus/pytorch:$PYTHONPATH
python -c "from nnunet.training.network_training.nnUNetPlusPlusTrainerV2_MLKA import nnUNetPlusPlusTrainerV2_MLKA; print('ok')"
```

### `RESULTS_FOLDER is not defined`

原因：nnU-Net 环境变量未设置。

处理：重新设置 `nnUNet_raw_data_base`、`nnUNet_preprocessed`、`RESULTS_FOLDER`。

### 推理找不到模型目录

检查训练和推理参数是否一致：

```text
训练: nnUNet_train 2d nnUNetPlusPlusTrainerV2_MLKA TaskXXX 0 -p nnUNetPlans
推理: nnUNet_predict -m 2d -tr nnUNetPlusPlusTrainerV2_MLKA -t TaskXXX -p nnUNetPlans
```
