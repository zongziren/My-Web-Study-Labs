## 环境

- Windows 10
- Intel(R) Core(TM) i7-9750H CPU @ 2.60GHz (12 CPUs), ~2.6GHz
- Python 3.9.9
- Requirment:

```
gensim==4.1.2
numpy==1.21.4
pandas==1.3.5
torch==1.10.1+cu113
torchaudio==0.10.1+cu113
torchkge==0.16.25
torchvision==0.11.2+cu113
tqdm==4.62.3
```

## TrabsE-Hand

- 手动完成（不依赖框架）的基于TrabsE算法的预测

### 1. 文件结构

The TransE-Hand includes the following files and directories:

```
├── data
│   ├── dev.txt
│   ├── test.txt
│   └── train.txt
├── out
│   └── out.txt
├── transe.py
├── dev.py
└── test.py
```

### 2. 运行方法

1. 执行 `python.exe transe.py` 生成 entity_100dim1 和 relation_100dim1
2. 执行 `python.exe dev.py` 可以验证目前的 hit
3. 执行 `python.exe test.py` 可以输出 out.txt 到 ou t文件夹下用来提交

## TrabsE-Hand-word2vec

- 在TrabsE-Hand的基础上利用word2vec使用文本信息来初始化

### 1. 文件结构

```
├── data
│   ├── dev.txt
│   ├── test.txt
│   └── train.txt
├── out
│   └── out.txt
├── transe.py
├── dev.py
└── test.py
```

### 2. 运行方法

1. 执行 `python.exe entity.py` 生成 entity_co.txt
2. 执行 `python.exe relation.py` 生成 relation_co.txt
3. 执行 `python.exe transe.py` 生成 entity_100dim1 和 relation_100dim1
4. 执行 `python.exe dev.py` 可以验证目前的 hit
5. 执行 `python.exe test.py` 可以输出 out.txt 到 out 文件夹下用来提交

## TransE-torchkge

- 使用 TorchKGE 库进行训练

### 1. 文件结构

```
TransE-TorchKGE
  ├─data
  │   ├── dev.txt
  │   ├── test.txt
  │   └── train.txt
  ├── test.py
  └── train.py
```

### 2. 使用方法

```sh
cd src/TransE-TorchKGE
python ./train.py
python ./test.py
```

`train.py` 训练 TransE 模型并在当前目录输出 `ent_100dim` 和 `rel_100dim`
`test.py` 对 `test.txt` 中尾实体进行预测，输出前五个结果到 `out.txt`
