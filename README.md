# MLP（纯 numpy 手写多层感知机）

用 numpy 从零实现的 MLP + 交叉熵训练，用于 MNIST 类图像分类，并支持把训练好的
模型参数导出为 `.npz` 文件供外部程序调用。

## 项目结构

```
model/
  Linear.py          全连接层（forward / backward / 参数）
  ReLU.py            ReLU 激活层
  CrossEntropyLoss.py  Softmax + 交叉熵损失（含梯度）
  Net.py             网络组装（save / load 参数导出/恢复）
train.py             训练脚本，结束后自动导出参数
visualize_predictions.py  随机抽样测试数据可视化预测，人工确认模型效果
```

依赖：Python 3 + numpy（训练还需 `data/train.csv`：首列为 0–9 标签，
其余为 784 个 0–255 像素值，与 MNIST CSV 格式一致）。

## 训练与导出

```bash
python train.py
```

训练结束后自动把模型参数保存到 `./model_params.npz`（可在 `train.py` 顶部
通过 `save_path` 修改路径），终端会打印 `Model parameters saved to ...`。

## 可视化抽查模型效果

```bash
python visualize_predictions.py                      # 每次运行固定随机抽 5 个样本
python visualize_predictions.py --seed 42            # 指定种子，复现同 5 个样本
python visualize_predictions.py --show               # 额外弹出 matplotlib 窗口
```

脚本每次运行固定随机抽取 5 个测试样本，用训练好的模型预测，并把 28×28
图像画成一张 `test_preview.png` 供人工确认：

- 不传 `--seed` 时每次运行都会重新随机抽 5 个不同样本；传 `--seed` 可复现；
- `data/test.csv` 没有标签列 → 每张图只标注预测类别与置信度，需要目视核对
  数字与预测是否一致；
- 若传入带 `label` 列的 CSV（如 `data/train.csv`），图题显示
  `pred / true`（正确绿色、错误红色），并打印整份数据的准确率；
- 其他参数：`--data`、`--model`、`--output`、`--seed`（可选）。

## 导出文件格式（.npz）

```python
import numpy as np
data = np.load('model_params.npz')
data.files  # ['layer_sizes', 'W0', 'b0', 'W1', 'b1', 'W2', 'b2']
```

| key            | shape              | 含义                                              |
|----------------|--------------------|---------------------------------------------------|
| `layer_sizes`  | (n_layers,)        | 各层神经元数，如 `[784, 128, 64, 10]`             |
| `W{i}`, `b{i}` | `(in, out)` / `(1, out)` | 第 i 个全连接层的权重与偏置（i 从 0 开始） |

第 i 个全连接层连接 `layer_sizes[i] → layer_sizes[i+1]`；除最后一层外，
每两个全连接层之间有一个 ReLU；输出层得到的是 logits，类别 = `argmax(logits)`。
偏置 `b` 在计算时按行广播即可（每行加上同一 `b[0]`）。

## 在外部程序中调用

### 方式一：使用本项目代码恢复网络（numpy 即可）

```python
import numpy as np
from model.Net import Net

net = Net.load('model_params.npz')          # 恢复完整网络
X = np.load('some_input.npy')               # shape (batch, 784)，需与训练时相同的预处理
logits = net.forward(X)                     # (batch, 10) logits
preds = np.argmax(logits, axis=1)           # (batch,) 每样本的类别
```

`Net.load` 返回的参数可写，也可继续 `net.backward(...)` / `net.step(lr)` 恢复训练。

### 方式二：不依赖本项目代码，纯 numpy 前向推理

```python
import numpy as np

data = np.load('model_params.npz')
sizes = [int(s) for s in data['layer_sizes']]
n_linear = len(sizes) - 1

def relu(x):
    return np.maximum(x, 0.0)

def predict(X):
    """X: (batch, sizes[0])，已按训练时的预处理归一化。"""
    h = X
    for i in range(n_linear):
        z = h @ data[f'W{i}'] + data[f'b{i}']
        h = relu(z) if i < n_linear - 1 else z   # 最后一层不加 ReLU
    return np.argmax(h, axis=1)
```

### 输入预处理（务必与训练一致）

`train.py` 中把 0–255 像素除以 255 得到 `[0, 1]`；若图片是 28×28，
需先展平为一维 784 再送入网络。外部程序传入前必须做同样的归一化。

## 备注

- 导出文件只包含可训练参数（权重/偏置），不含激活层结构——结构完全由
  `layer_sizes` 决定；若日后修改 `Net` 的层组装规则，请同步导出/恢复逻辑。
- 权重在 `train.py` 中以 0.01 系数初始化；本仓库未保存优化器状态
  （当前算法就是普通 SGD，恢复后直接继续 `step` 即可）。
