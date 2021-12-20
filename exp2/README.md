## 环境

- Windows 11
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

### 文件结构

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

### 运行方法

1. 执行 `python.exe transe.py` 生成 entity_100dim1 和 relation_100dim1
2. 执行 `python.exe dev.py` 可以验证目前的 hit情况
3. 执行 `python.exe test.py` 可以输出 out.txt 到 ou t文件夹下用来提交

### 关键函数

- `transe.py`
    - `data_loader` 函数读取 `train.txt` 数据集，获取 `entity_set`, `relation_set`, `triple_list` 列表
    - `def distanceL` 函数求梯度
    - `emb_initialize`函数对向量使用随机数进行初始化
    - `train`函数进行训练
      - embedding_dim=100, learning_rate=0.01, margin=1, L1=True
      - transE.train(epochs=2000) 2000次迭代
- `test.py`
    - `data_loader` 函数读取数据集
    - `transE_loader` 函数读取 `ent` 和 `rel` 向量
    - `distance` 函数根据 TransE 模型计算距离
    - `mean_rank` 函数根据计算距离，选择前五个输出

## TrabsE-Hand-word2vec

- 在TrabsE-Hand的基础上利用word2vec使用文本信息来初始化

### 文件结构

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

### 运行方法

1. 执行 `python.exe entity.py` 生成 entity_co.txt
2. 执行 `python.exe relation.py` 生成 relation_co.txt
3. 执行 `python.exe transe.py` 生成 entity_100dim1 和 relation_100dim1
4. 执行 `python.exe dev.py` 可以验证目前的 hit
5. 执行 `python.exe test.py` 可以输出 out.txt 到 out 文件夹下用来提交

### 关键函数

- `transe.py`
    - 调用word2vec进行训练。
     ```
    model_relation = word2vec.Word2Vec(sentences, 
    hs=1, min_count=1, window=3, size=100)
    model_entity = word2vec.Word2Vec(sentences,
    hs=1, min_count=1, window=3, size=100)
    ```
    - `data_loader` 函数读取 `train.txt` 数据集，获取 `entity_set`, `relation_set`, `triple_list` 列表
    - `def distanceL` 函数求梯度
    - `emb_initialize`函数对读取word2vec生成的向量进行初始化
    - `train`函数进行训练
      - embedding_dim=100, learning_rate=0.01, margin=1, L1=True
      - transE.train(epochs=2000) 2000次迭代
- `test.py`
    - `data_loader` 函数读取数据集
    - `transE_loader` 函数读取 `ent` 和 `rel` 向量
    - `distance` 函数根据 TransE 模型计算距离
    - `mean_rank` 函数根据计算距离，选择前五个输出

## TransE-torchkge

使用 TorchKGE 库进行训练

### 文件结构

```
TransE-TorchKGE
  ├─data
  │   ├── dev.txt
  │   ├── test.txt
  │   └── train.txt
  ├── test.py
  └── train.py
```

### 使用方法

```sh
cd src/TransE-TorchKGE
python ./train.py
python ./test.py
```

需对生成的 `out.txt` 进行处理，去除每一行末尾的 `,` 并去除文件尾部空行

`train.py` 训练 TransE 模型并在当前目录输出 `ent_100dim` 和 `rel_100dim`
`test.py` 对 `test.txt` 中尾实体进行预测，输出前五个结果到 `out.txt`

### 关键函数

- `train.py`
    - `load_dataset` 函数读取 `train.txt`、`dev.txt` 和 `test.txt` 数据集，获取 `ent` 和 `rel` 列表
    - `main` 函数设置模型参数，进行训练，并输出结果到 `ent_100dim` 和 `rel_100dim`
- `test.py`
    - `data_loader` 函数读取数据集
    - `transE_loader` 函数读取 `ent` 和 `rel` 向量
    - `distance` 函数根据 TransE 模型计算距离
    - `mean_rank` 函数根据计算距离，选择前五个输出
