# YOLOv8 ONNX 模型转换为 RKNN

本文说明如何在 Ubuntu 环境中使用 RKNN-Toolkit2，将自定义的 YOLOv8 ONNX 模型转换为适用于 RK3566 的 INT8 RKNN 模型。

## 1. 准备 Python 环境

建议使用 Conda 创建独立的 Python 3.8 环境：

```bash
conda create -n rknn-toolkit2 python=3.8 -y
conda activate rknn-toolkit2

python -m pip install --upgrade pip
cd /path/to/rknn_model_zoo-main
pip install -r docs/requirements_cp38.txt
```

从 [rknn-toolkit2](https://github.com/airockchip/rknn-toolkit2) 获取与 Python 3.8、Ubuntu 和当前 CPU 架构匹配的 RKNN-Toolkit2 wheel，然后安装。例如：

```bash
pip install /path/to/rknn_toolkit2-<version>-cp38-cp38-linux_x86_64.whl
```

安装完成后验证环境：

```bash
python -c "from rknn.api import RKNN; print('RKNN-Toolkit2 is ready')"
```

> RK3566 属于 RKNPU2 平台，应使用 **RKNN-Toolkit2**。wheel 文件名和版本以实际下载的 SDK 为准。

## 2. 放置 ONNX 模型

将训练并导出的 ONNX 模型放入以下目录：

```text
rknn_model_zoo-main/examples/yolov8/model/yolov8n.onnx
```

文件名可以自定义，但后续转换命令中的输入路径必须与其保持一致。

## 3. 创建 INT8 量化数据集列表

INT8 转换需要使用有代表性的图片进行量化校准。建议从实际训练集或部署场景中选取约 20 张图片，并创建文件：

```text
rknn_model_zoo-main/examples/yolov8/model/dataset.txt
```

该文件每行填写一张图片的路径，推荐使用绝对路径。例如：

```text
/home/user/datasets/my_dataset/images/train/image_001.jpg
/home/user/datasets/my_dataset/images/train/image_002.jpg
/home/user/datasets/my_dataset/images/train/image_003.jpg
/home/user/datasets/my_dataset/images/train/image_004.jpg
/home/user/datasets/my_dataset/images/train/image_005.jpg
# 继续添加，合计约 20 张；实际文件中不要保留本行注释
```

图片应覆盖常见目标、光照、背景和尺寸等部署场景。这里只需要原始图片，不需要标签文件。

## 4. 修改转换脚本

打开：

```text
examples/yolov8/python/convert.py
```

将第 4 行的量化数据集路径修改为刚创建的 `dataset.txt`：

```python
DATASET_PATH = '../model/dataset.txt'
```

该相对路径以运行命令时所在的 `examples/yolov8/python` 目录为基准。如果将 `dataset.txt` 放在其他位置，请相应修改此变量；也可以直接填写绝对路径。

## 5. 执行转换

进入转换脚本所在目录并运行：

```bash
cd /path/to/rknn_model_zoo-main/examples/yolov8/python
python convert.py ../model/yolov8n.onnx rk3566 i8 ../model/yolov8.rknn
```

参数含义：

| 参数 | 说明 |
| --- | --- |
| `../model/yolov8n.onnx` | 输入的 ONNX 模型路径 |
| `rk3566` | 目标芯片平台 |
| `i8` | 执行 INT8 量化，转换时会读取 `DATASET_PATH` |
| `../model/yolov8.rknn` | 输出的 RKNN 模型路径 |

当终端依次显示 `Config model`、`Loading model`、`Building model` 和 `Export rknn model` 均完成后，模型将输出到：

```text
rknn_model_zoo-main/examples/yolov8/model/yolov8.rknn
```

可使用以下命令确认文件已经生成：

```bash
ls -lh ../model/yolov8.rknn
```

## 注意事项

- 转换必须在安装了 RKNN-Toolkit2 的 Ubuntu 主机环境中执行，不是在 RK3566 开发板上执行。
- 自定义 ONNX 模型的输入尺寸、输出节点和后处理方式需要与本项目的 YOLOv8 示例兼容。
- 若不需要量化，可将命令中的 `i8` 改为 `fp`；FP 模式不会使用量化图片列表。
- 部署时应确保板端驱动和 RKNN Runtime 与 RKNN-Toolkit2 版本兼容。
