# 中文命名实体识别（NER）

本项目实现了多种序列标注模型（HMM、CRF、Bi-LSTM、Bi-LSTM+CRF）来完成中文命名实体识别。数据集采用 ACL 2018 论文 [Chinese NER using Lattice LSTM](https://github.com/jiesutd/LatticeLSTM) 中的简历数据，位于 `DataNER` 目录下，格式为每行一个字符及其 BIOES 标注，句子之间用空行分隔。

## 环境依赖

安装依赖：

```bash
pip3 install -r requirement.txt
```

## 项目结构

- `DataNER/`：训练、验证、测试集（`.char.bmes` 格式）。
- `models/`：各模型实现与配置，其中 `config.py` 控制模型及训练超参。
- `utils.py`、`data.py`：数据读取、特征处理与工具函数。
- `main.py`：命令行入口，按顺序训练并评估全部模型。
- `test.py`：加载已训练模型进行统一评估。
- `output.txt`：作者本地测评时的预期输出示例。
- `gui_app/`：图形界面代码，`main.py` 为 GUI 启动入口。

## 使用指南

### 训练与评估（命令行）
- 在项目根目录运行 `python3 main.py` 可依次训练并评估 HMM、CRF、Bi-LSTM 与 Bi-LSTM+CRF，全流程会重新训练模型并输出各指标。
- 训练参数、模型配置可在 `models/config.py` 中调整。

### 仅评估已训练模型
- 运行 `python3 test.py` 会加载现有权重对数据集进行统一测评。
- `output.txt` 保存了作者测评时的预期输出，可用于对照验证。

### 图形化界面
- 在 `gui_app` 目录下执行 `python3 main.py` 启动用户界面。
- 启动成功后会弹出窗口，可选择特定模型执行训练或评估，并可输入自定义句子体验模型标注效果。

