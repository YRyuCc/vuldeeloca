# VulDee — VSCode 漏洞检测插件

基于论文 *VulDeeLocator: A Deep Learning-Based Fine-Grained Vulnerability
Detector* 的原理，用当前主流库（PyTorch、gensim、tree-sitter）重新实现的
C/C++ 漏洞检测与定位工具。它既能检测"有没有漏洞"，也能把漏洞定位到具体行，
并以 VSCode 波浪线的方式展示。

## 总体架构

项目分成两个部分：

- `src/` — VSCode 扩展（TypeScript），负责界面、触发扫描、把结果画成诊断波浪线。
- `backend/` — 检测引擎（Python 包 `vuldee`），负责全部算法。

扩展不内嵌模型，而是调用 Python 后端的命令行接口
（`python -m vuldee.cli scan ...`）拿到 JSON 结果。这样后端可以独立训练、独立测试。

## 后端模块划分

所有功能都拆成了独立模块，与原项目一一对应：

| 模块 | 职责 | 对应的原实现 |
| --- | --- | --- |
| `vuldee.source2slice.focus` | 四类关注点提取（FC/AD/PD/AE） | `ngetapi/ngetarr/ngetpoint/ngetval` |
| `vuldee.source2slice.slicer` | LLVM IR 反向程序切片 | `dg` 的 `llvm-slicer` |
| `vuldee.source2slice.ir` | clang 编译 + 调试行号映射 + 焦点切片 | `get-llvmwithline` + `nslicerline` |
| `vuldee.ingest.sard` | 解析 SARD XML + clang 语义切片成语料 | `getVulLineForCounting.py` + `multiFileCompile.py` |
| `vuldee.ingest.nvd` | NVD vul/patch 对 diff + 语义切片成语料 | NVD `sourcefile` 工具链 |
| `vuldee.data.tokenizer` | 分词 + 符号抽象（func_N/variable_N） | `get_tokens.py` + `mapping.py` |
| `vuldee.data.corpus` | 切片转语料（行号/漏洞行索引） | `process_dataflow.py` |
| `vuldee.data.word2vec` | 训练词向量 | `create_word2vecmodel.py` |
| `vuldee.data.vectorize` | token 转向量、定长窗口 | `get_dl_input.py` |
| `vuldee.data.dataset` | PyTorch 数据集/批处理 | `preprocess_dl_Input_version4.py` |
| `vuldee.models.bgru` | 双向 GRU 检测模型 | `bgru.py` |
| `vuldee.models.bgru_locator` | 检测 + 定位模型 | `bgru_threshold.py` |
| `vuldee.models.metrics` | TP/FP/FN/精确率/召回率/F1 | `metrics.py` |
| `vuldee.train` / `vuldee.infer` | 训练/评估与推理 | 两个模型文件的 `main` |
| `vuldee.cli` | 命令行入口 | — |

## 环境安装

前置要求：Python 3.10+（推荐 3.12）、Node.js 18+。可选：本机安装
LLVM/clang（只有做完整语义切片时才需要，纯源码扫描不需要）。

推荐用 conda（或 venv）建一个隔离环境，二选一即可：

```bash
# conda 方式
conda env create -f environment.yml
conda activate vuldee
pip install -e backend

# venv 方式（用你本机的 Python 3.10+）
python -m venv .venv
# Windows: .venv\Scripts\activate   macOS/Linux: source .venv/bin/activate
pip install -r backend/requirements.txt
pip install -e backend
```

## 快速体验（无需任何数据）

说明：仓库里**不包含训练好的模型文件**（模型较大，且会带上本地路径，已加入
`.gitignore`），所以下载后需要先跑下面的 `demo` 生成一个演示模型，或者用
真实语料自己训练，之后才能 `scan`。

先生成一小份合成数据并训练一个演示模型（约几十秒）：

```bash
vuldee demo --out-dir models/demo --epochs 5
```

然后扫描任意 C/C++ 文件：

```bash
vuldee scan --file path/to/file.c --model-dir models/demo
```

输出是 JSON，包含每个可疑行号、关注点类型、函数名和置信度。

## 用真实数据训练

先解压原始语料，再用 clang 做完整语义切片生成语料：

```bash
# SARD（需要 SARD_testcaseinfo.xml 和 synthetic and academic programs 目录）
vuldee ingest-sard --programs-root data/sard --xml SARD_testcaseinfo.xml \
  --out corpus_sard.pkl --limit 300 --testcase-lib path/to/testcaseLib

# NVD（需要 real-world programs 目录，vul/patch 成对文件）
vuldee ingest-nvd --programs-root data/nvd --out corpus_nvd.pkl --limit 200

# 训练词向量 + 模型
vuldee w2v --corpus corpus_sard.pkl --out word2vec.model
vuldee train --corpus corpus_sard.pkl --w2v word2vec.model --out-dir models/mymodel --epochs 4
```

`ingest-sard`/`ingest-nvd` 会对每个 C/C++ 文件调用 clang 生成带调试行号的
LLVM IR，按四类关注点做反向程序切片，并用漏洞行标注生成训练样本；文件无法
编译时自动回退到源码窗口切片（NVD 真实项目文件在 Windows 上大多无法独立
编译，故主要走该回退路径）。

## 在 VSCode 里使用

1. 安装依赖：`npm install && npm run compile`
2. 用 VSCode 打开本目录，按 F5 启动扩展开发宿主
3. 在设置里配好两个参数：
   - `vuldee.pythonPath`：装有 `vuldee` 包的解释器路径（留空则用 PATH 里的 `python`）
   - `vuldee.modelDir`：模型目录（含 `locator.pt`、`word2vec.model`、`config.json`）
4. 打开一个 C/C++ 文件，运行命令面板里的 `VulDee: Scan current file`
5. 可选：把 `vuldee.scanOnSave` 设为 `true`，保存时自动扫描

命令面板里的 `VulDee: Open training & settings panel` 会打开一个图形面板：
在里面填写 Python 解释器、模型目录、语料/词向量路径和训练超参，点击按钮即可
保存配置、生成演示模型、在真实语料上训练，或扫描当前文件；训练日志会实时
滚动显示。

## 运行测试

```bash
cd backend
pytest ../tests
```

## 与原项目的差异与说明

- 原实现依赖 TensorFlow 1.x / Keras 2.1.2，这里用 PyTorch 重写，API 全部现代化。
- 原来的 libclang 访客和 `dg` 切片器是 Linux 专属工具链；这里关注点提取改用
  `tree-sitter`（纯 pip、跨平台），切片改为纯 Python 的 LLVM IR 反向数据流切片。
- 模型结构与原论文保持一致：两层双向 GRU + 逐 token 的 sigmoid 头，训练目标
  是"漏洞 token 掩码 × 概率"的 k-max，定位用逐行 top-k 阈值规则。
- 若要用完整语义切片（需要 clang 生成 LLVM IR），安装 LLVM 后即可接入
  `vuldee.source2slice.slicer`；开箱即用的扫描走源码级候选窗口，无需 clang。
