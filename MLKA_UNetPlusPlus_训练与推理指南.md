# MLKA UNet++ 训练与推理指南

本文档说明如何在 GPU 服务器上使用当前 UNet++ 改动版本，包括 `nnUNetPlusPlusTrainerV2_MLKA`、`nnUNetPlusPlusTrainerV2_GSAU` 与 `nnUNetPlusPlusTrainerV2_MLKA_GSAU`。

## 0. 快速上手

这一节按“直接跑程序”的顺序写。第一次使用时，先把下面几个变量替换成你自己的路径和任务名，然后逐段执行。

### 0.1 先替换这些变量

```bash
export REPO=/path/to/New_UNetPlusPlus
export RAW_BASE=/data/nnUNet_raw_data_base
export PREPROCESSED=/data/nnUNet_preprocessed
export RESULTS=/data/nnUNet_results
export TASK_ID=101
export TASK_NAME=Task101_RoadSeg
export GPU=0
```

含义如下：

| 变量 | 示例 | 含义 |
|------|------|------|
| `REPO` | `/path/to/New_UNetPlusPlus` | 当前仓库根目录 |
| `RAW_BASE` | `/data/nnUNet_raw_data_base` | 原始 Task 数据根目录 |
| `PREPROCESSED` | `/data/nnUNet_preprocessed` | 预处理与 plans 输出目录 |
| `RESULTS` | `/data/nnUNet_results` | checkpoint、日志、验证结果目录 |
| `TASK_ID` | `101` | nnU-Net 任务编号 |
| `TASK_NAME` | `Task101_RoadSeg` | 任务文件夹名 |
| `GPU` | `0` | 使用的 GPU 编号 |

### 0.2 安装环境

```bash
cd $REPO
conda create -n unetpp_mlka python=3.9 -y
conda activate unetpp_mlka
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements-unetpp.txt
pip install --no-deps -e ./pytorch
```

如果服务器 CUDA 不是 11.8，把 `cu118` 换成对应版本，例如 `cu121`。安装后确认 GPU 可用：

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

### 0.3 设置 nnU-Net 路径

```bash
export nnUNet_raw_data_base=$RAW_BASE
export nnUNet_preprocessed=$PREPROCESSED
export RESULTS_FOLDER=$RESULTS
export PYTHONPATH=$REPO/pytorch:$PYTHONPATH
```

Windows PowerShell 对应写法：

```powershell
$env:nnUNet_raw_data_base="D:\data\nnUNet_raw_data_base"
$env:nnUNet_preprocessed="D:\data\nnUNet_preprocessed"
$env:RESULTS_FOLDER="D:\data\nnUNet_results"
$env:PYTHONPATH="D:\path\to\New_UNetPlusPlus\pytorch;$env:PYTHONPATH"
```

### 0.4 准备数据

你的数据必须放成下面的 nnU-Net Task 格式。仓库本身没有可直接复现训练的实际数据。

```text
$RAW_BASE/nnUNet_raw_data/$TASK_NAME/
├── dataset.json
├── imagesTr/
│   ├── case001_0000.nii.gz
│   └── ...
├── labelsTr/
│   ├── case001.nii.gz
│   └── ...
└── imagesTs/
    ├── case101_0000.nii.gz
    └── ...
```

单模态图像用 `_0000.nii.gz`；多模态继续 `_0001.nii.gz`、`_0002.nii.gz`。标签文件不带 `_0000`，必须是整数类别，背景为 `0`。

### 0.5 规划与预处理

当前含 MLKA 的实验必须使用 `base_num_features=30`，因此使用 `ExperimentPlanner2D`：

```bash
nnUNet_plan_and_preprocess -t $TASK_ID -pl2d ExperimentPlanner2D -pl3d None --verify_dataset_integrity
```

确认 plans 正确：

```bash
python -c "from batchgenerators.utilities.file_and_folder_operations import load_pickle; import os; p=load_pickle(os.path.join(os.environ['nnUNet_preprocessed'], '$TASK_NAME', 'nnUNetPlans_plans_2D.pkl')); print('base_num_features =', p['base_num_features'])"
```

必须看到：

```text
base_num_features = 30
```

### 0.6 先跑一个 fold 做冒烟训练

先跑第 `0` 折确认链路通畅。四组实验对应 trainer 如下：

| 实验组 | Trainer |
|------|------|
| Baseline | `nnUNetPlusPlusTrainerV2` |
| MLKA only | `nnUNetPlusPlusTrainerV2_MLKA` |
| GSAU only | `nnUNetPlusPlusTrainerV2_GSAU` |
| MLKA + GSAU | `nnUNetPlusPlusTrainerV2_MLKA_GSAU` |

```bash
CUDA_VISIBLE_DEVICES=$GPU nnUNet_train 2d nnUNetPlusPlusTrainerV2 $TASK_NAME 0 -p nnUNetPlans
CUDA_VISIBLE_DEVICES=$GPU nnUNet_train 2d nnUNetPlusPlusTrainerV2_MLKA $TASK_NAME 0 -p nnUNetPlans
CUDA_VISIBLE_DEVICES=$GPU nnUNet_train 2d nnUNetPlusPlusTrainerV2_GSAU $TASK_NAME 0 -p nnUNetPlans
CUDA_VISIBLE_DEVICES=$GPU nnUNet_train 2d nnUNetPlusPlusTrainerV2_MLKA_GSAU $TASK_NAME 0 -p nnUNetPlans
```

训练输出会分别进入：

```text
$RESULTS/nnUNet/2d/$TASK_NAME/<TrainerName>__nnUNetPlans/fold_0/
```

### 0.7 正式 5 折训练

正式消融建议 5 折都跑，并保持四组完全相同的数据、plans、fold、TTA 设置。

```bash
for FOLD in 0 1 2 3 4; do
  CUDA_VISIBLE_DEVICES=$GPU nnUNet_train 2d nnUNetPlusPlusTrainerV2 $TASK_NAME $FOLD -p nnUNetPlans
  CUDA_VISIBLE_DEVICES=$GPU nnUNet_train 2d nnUNetPlusPlusTrainerV2_MLKA $TASK_NAME $FOLD -p nnUNetPlans
  CUDA_VISIBLE_DEVICES=$GPU nnUNet_train 2d nnUNetPlusPlusTrainerV2_GSAU $TASK_NAME $FOLD -p nnUNetPlans
  CUDA_VISIBLE_DEVICES=$GPU nnUNet_train 2d nnUNetPlusPlusTrainerV2_MLKA_GSAU $TASK_NAME $FOLD -p nnUNetPlans
done
```

5 折完成后，为每个实验组汇总后处理配置：

```bash
nnUNet_determine_postprocessing -m 2d -t $TASK_NAME -tr nnUNetPlusPlusTrainerV2 -pl nnUNetPlans -val validation_raw
nnUNet_determine_postprocessing -m 2d -t $TASK_NAME -tr nnUNetPlusPlusTrainerV2_MLKA -pl nnUNetPlans -val validation_raw
nnUNet_determine_postprocessing -m 2d -t $TASK_NAME -tr nnUNetPlusPlusTrainerV2_GSAU -pl nnUNetPlans -val validation_raw
nnUNet_determine_postprocessing -m 2d -t $TASK_NAME -tr nnUNetPlusPlusTrainerV2_MLKA_GSAU -pl nnUNetPlans -val validation_raw
```

### 0.8 推理

准备待预测数据：

```text
/data/inference/imagesTs/
├── case101_0000.nii.gz
├── case102_0000.nii.gz
└── ...
```

推理四组模型：

```bash
export INFER_IN=/data/inference/imagesTs
export INFER_OUT=/data/inference/$TASK_NAME

CUDA_VISIBLE_DEVICES=$GPU nnUNet_predict -i $INFER_IN -o $INFER_OUT/pred_baseline -t $TASK_NAME -m 2d -tr nnUNetPlusPlusTrainerV2 -p nnUNetPlans -f 0
CUDA_VISIBLE_DEVICES=$GPU nnUNet_predict -i $INFER_IN -o $INFER_OUT/pred_mlka -t $TASK_NAME -m 2d -tr nnUNetPlusPlusTrainerV2_MLKA -p nnUNetPlans -f 0
CUDA_VISIBLE_DEVICES=$GPU nnUNet_predict -i $INFER_IN -o $INFER_OUT/pred_gsau -t $TASK_NAME -m 2d -tr nnUNetPlusPlusTrainerV2_GSAU -p nnUNetPlans -f 0
CUDA_VISIBLE_DEVICES=$GPU nnUNet_predict -i $INFER_IN -o $INFER_OUT/pred_mlka_gsau -t $TASK_NAME -m 2d -tr nnUNetPlusPlusTrainerV2_MLKA_GSAU -p nnUNetPlans -f 0
```

如果已经完成 5 折训练，可去掉 `-f 0`，nnU-Net 会自动使用已有 folds 做集成。输出目录中最重要的是每个 case 的 `.nii.gz` 分割 mask。

### 0.9 最小检查清单

跑不起来时，优先检查这 6 项：

1. 三个环境变量是否存在：`nnUNet_raw_data_base`、`nnUNet_preprocessed`、`RESULTS_FOLDER`。
2. 数据是否在 `$RAW_BASE/nnUNet_raw_data/$TASK_NAME/`。
3. 图像命名是否是 `caseID_0000.nii.gz`，标签是否是 `caseID.nii.gz`。
4. `nnUNetPlans_plans_2D.pkl` 的 `base_num_features` 是否等于 `30`。
5. 训练和推理的 `-m 2d`、`-tr`、`-p nnUNetPlans` 是否一致。
6. 推理输入目录是否只放待预测图像，不要把标签、结果或其他文件混进去。

## 1. 当前支持范围

| 项目 | 要求 |
|------|------|
| 网络 | 仅 `2d` |
| Trainer | `nnUNetPlusPlusTrainerV2_MLKA` / `nnUNetPlusPlusTrainerV2_GSAU` / `nnUNetPlusPlusTrainerV2_MLKA_GSAU` |
| plans | 主实验和所有含 MLKA 的实验必须使用 `base_num_features=30` 的 2D plans |
| 上采样 | 必须 `convolutional_upsampling=True` |
| 拓扑 | 当前 `Generic_UNetPlusPlus.forward()` 固定 5-pool |
| 不支持 | 3D、含 MLKA 时的 `base_num_features=32`、非卷积上采样 |

不要直接使用默认 `nnUNetPlansv2.1` 的 2D plans。该 planner 默认 `base_num_features=32`，当前 MLKA 会报错。

GSAU-only 在结构上可以使用 `base_num_features=32`，但四组消融实验应统一使用同一份 `base_num_features=30` plans，否则 Baseline、MLKA、GSAU、MLKA+GSAU 的差异会混入通道数变化，不利于公平比较。

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

当前仓库只包含 `pytorch/nnunet/dataset_conversion/` 下的公开数据集转换脚本，不包含可直接训练复现的实际数据文件。也就是说，仓库内没有现成的：

```text
dataset.json
imagesTr/
labelsTr/
imagesTs/
*.nii.gz
*.npz
```

因此训练复现必须满足二者之一：

1. 使用你自己的数据，并整理成 nnU-Net 标准 Task 格式。
2. 从对应公开数据集下载原始数据，再使用仓库中的 `dataset_conversion/TaskXXX_*.py` 脚本转换。

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

预处理完成后，建议检查以下关键产物是否存在：

```text
$nnUNet_raw_data_base/nnUNet_cropped_data/TaskXXX_TaskName/
├── dataset.json
├── dataset_properties.pkl
├── gt_segmentations/
└── case001.npz / case001.pkl / ...

$nnUNet_preprocessed/TaskXXX_TaskName/
├── dataset.json
├── dataset_properties.pkl
├── nnUNetPlans_plans_2D.pkl
├── gt_segmentations/
└── nnUNet_2D_stage0/
    ├── case001.npz
    ├── case001.pkl
    └── ...
```

训练时实际读取的是 `$nnUNet_preprocessed/TaskXXX_TaskName/nnUNet_2D_stage0/` 下的预处理 `.npz/.pkl`。首次训练会把 `.npz` 解包成 `.npy` 以加快读取，这是正常现象。

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

### 6.5 训练后的验证与后处理

`nnUNet_train` 在训练结束后会自动运行一次 validation，并把验证集预测写到当前 fold 目录下。若只想基于已有 checkpoint 重新验证，可运行：

```bash
CUDA_VISIBLE_DEVICES=0 nnUNet_train 2d nnUNetPlusPlusTrainerV2_MLKA TaskXXX_TaskName 0 -p nnUNetPlans -val --npz
```

`--npz` 会额外保存 softmax 概率，适合后续 ensemble 或更细的误差分析；只看 Dice 和 mask 时可以不加。

如果已经完成 5 折训练，建议在每个实验组训练完后汇总后处理配置：

```bash
nnUNet_determine_postprocessing -m 2d -t TaskXXX_TaskName -tr nnUNetPlusPlusTrainerV2_MLKA -pl nnUNetPlans -val validation_raw
```

该命令会在 trainer 根目录生成推理阶段可自动使用的 `postprocessing.json`。如果只训练了单个 fold，默认汇总命令会因为缺少其他 folds 而失败；此时应明确报告“单 fold raw/postprocessed validation”，或在所有实验组都采用同一策略后，再考虑把 `fold_0/postprocessing.json` 复制到 trainer 根目录用于测试集推理。

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
$RESULTS_FOLDER/nnUNet/2d/TaskXXX_TaskName/<TrainerName>__nnUNetPlans/
```

四组消融对应目录：

```text
$RESULTS_FOLDER/nnUNet/2d/TaskXXX_TaskName/nnUNetPlusPlusTrainerV2__nnUNetPlans/
$RESULTS_FOLDER/nnUNet/2d/TaskXXX_TaskName/nnUNetPlusPlusTrainerV2_MLKA__nnUNetPlans/
$RESULTS_FOLDER/nnUNet/2d/TaskXXX_TaskName/nnUNetPlusPlusTrainerV2_GSAU__nnUNetPlans/
$RESULTS_FOLDER/nnUNet/2d/TaskXXX_TaskName/nnUNetPlusPlusTrainerV2_MLKA_GSAU__nnUNetPlans/
```

每个 trainer 下按 fold 保存：

```text
fold_0/
├── model_final_checkpoint.model
├── model_final_checkpoint.model.pkl
├── model_best.model
├── model_best.model.pkl
├── debug.json
├── progress.png
├── postprocessing.json
├── validation_raw/
├── validation_raw_postprocessed/
└── training_log_*.txt
```

trainer 根目录通常还会有 `plans.pkl`。如果做了 5 折后处理汇总，还会有 `postprocessing.json`、`cv_niftis_raw/` 和 `cv_niftis_postprocessed/`。推理脚本会从 trainer 根目录读取 `plans.pkl`，并优先使用 trainer 根目录的 `postprocessing.json`；如果根目录没有 `postprocessing.json`，推理不会中断，但会提示缺少后处理并输出 raw mask。

验证输出通常位于：

```text
$RESULTS_FOLDER/nnUNet/2d/TaskXXX_TaskName/<TrainerName>__nnUNetPlans/fold_0/validation_raw/
```

常见内容：

```text
validation_raw/
├── validation_args.json
├── summary.json
├── case001.nii.gz
├── case002.nii.gz
└── ...
```

如果验证时使用 `--npz`，同一目录下还会出现 `case001.npz` 和 `case001.pkl`。`summary.json` 是主要的定量结果文件；`case*.nii.gz` 是验证集分割 mask；`validation_raw_postprocessed/summary.json` 是经过该 fold 后处理策略后的验证指标。

建议手动推理输出按实验组分目录，避免覆盖：

```text
/data/inference/
├── pred_baseline/
│   ├── case101.nii.gz
│   └── ...
├── pred_mlka/
│   ├── case101.nii.gz
│   └── ...
├── pred_gsau/
│   ├── case101.nii.gz
│   └── ...
└── pred_mlka_gsau/
    ├── case101.nii.gz
    └── ...
```

如果使用 `-z/--save_npz`，同一推理输出目录下还会生成对应 `.npz` softmax 概率文件，可用于 ensemble。

推理输出目录通常应包含：

```text
pred_mlka/
├── plans.pkl
├── postprocessing.json        # 仅当 trainer 根目录存在后处理配置时自动复制
├── case101.nii.gz             # 最终分割 mask，标签值应与 dataset.json 的 labels 对应
├── case102.nii.gz
├── case101.npz / case101.pkl  # 仅在使用 -z/--save_npz 时生成
└── ...
```

论文统计优先使用验证或测试集的 `summary.json`、每类 Dice/HD 等指标，以及最终 `.nii.gz` mask。`progress.png`、`training_log_*.txt`、checkpoint 和 `.npz` softmax 属于训练追踪或中间产物，不应直接当作最终分割结果。

## 9. 限制与注意事项

1. 只能训练 `2d`，不要用 `3d_fullres`、`3d_lowres`、`3d_cascade_fullres`。
2. 必须使用 `-p nnUNetPlans`，前提是该 plans 由 `ExperimentPlanner2D` 生成，且 `base_num_features=30`。
3. 不要用默认 `nnUNetPlansv2.1` 训练 MLKA，2D v2.1 默认 `base_num_features=32`，当前不支持。
4. 当前 MLKA 固定 3 分支，要求节点通道能被 3 整除且每组不少于 8 通道。
5. 当前支持的主实验配置是 `base_num_features=30`，对应通道 `30, 60, 120, 240, 480`。
6. 推理时的 `-tr` 和 `-p` 必须与训练时一致。
7. 原始 baseline 使用 `nnUNetPlusPlusTrainerV2`；MLKA 使用 `nnUNetPlusPlusTrainerV2_MLKA`，不要混用结果目录。
8. 新增 trainer 文件 `nnUNetPlusPlusTrainerV2_MLKA.py`、`nnUNetPlusPlusTrainerV2_GSAU.py`、`nnUNetPlusPlusTrainerV2_MLKA_GSAU.py` 必须同步到 GPU 服务器。

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

原因：对应 trainer 文件没有同步到 GPU 服务器，或 `PYTHONPATH` 没指向当前仓库。

处理：

```bash
export PYTHONPATH=/path/to/New_UNetPlusPlus/pytorch:$PYTHONPATH
python -c "from nnunet.training.network_training.nnUNetPlusPlusTrainerV2_MLKA import nnUNetPlusPlusTrainerV2_MLKA; from nnunet.training.network_training.nnUNetPlusPlusTrainerV2_GSAU import nnUNetPlusPlusTrainerV2_GSAU; from nnunet.training.network_training.nnUNetPlusPlusTrainerV2_MLKA_GSAU import nnUNetPlusPlusTrainerV2_MLKA_GSAU; print('ok')"
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

## 11. 完整链路自检结论

当前仓库的完整链路是可运行的，但仓库本身不包含可直接复现实验的数据集。你必须先提供符合 nnU-Net Task 规范的数据，或下载公开数据后用 `pytorch/nnunet/dataset_conversion/TaskXXX_*.py` 转换。

从数据到结果的实际路径如下：

```text
原始 Task 数据
$nnUNet_raw_data_base/nnUNet_raw_data/TaskXXX_TaskName/
    -> nnUNet_plan_and_preprocess
裁剪与预处理
$nnUNet_raw_data_base/nnUNet_cropped_data/TaskXXX_TaskName/
$nnUNet_preprocessed/TaskXXX_TaskName/nnUNet_2D_stage0/
    -> nnUNet_train
训练、验证与 checkpoint
$RESULTS_FOLDER/nnUNet/2d/TaskXXX_TaskName/<TrainerName>__nnUNetPlans/fold_0/
    -> nnUNet_predict
测试/推理 mask
/data/inference/pred_<experiment_name>/
```

复现实验时建议逐项确认：

1. `dataset.json` 中的 `modality` 数量必须等于每个 case 的 `_0000/_0001/...` 文件数量。
2. `labelsTr/caseID.nii.gz` 的空间尺寸、spacing、方向应与 `imagesTr/caseID_0000.nii.gz` 对齐。
3. 标签值必须是整数，且与 `dataset.json` 的 `labels` 对应；背景为 `0`。
4. `nnUNetPlans_plans_2D.pkl` 的 `base_num_features` 必须为 `30`，否则不要训练 MLKA 或 MLKA+GSAU。
5. 四组消融必须使用同一 Task、同一 plans、同一 fold、同一训练轮数和同一推理 TTA 设置。
6. 推理目录中每个 case 必须按训练模态数完整提供，例如单模态为 `case101_0000.nii.gz`，四模态为 `case101_0000.nii.gz` 到 `case101_0003.nii.gz`。

预期结果形态不是“一个图片文件”，而是一组可追踪产物：训练日志和曲线、模型 checkpoint、验证集 `.nii.gz` mask、验证 `summary.json`，以及独立推理目录下的测试 `.nii.gz` mask。真正支撑论文结论的是四组消融在相同数据划分上的 `summary.json` 指标和对应 mask，可视化图只作为辅助展示。
