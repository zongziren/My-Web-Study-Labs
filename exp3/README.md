## model_A：BPR

### 环境

#### C++ 部分

- OS: Arch Linux on Windows 10 x86_64
- Kernel: 5.10.60.1-microsoft-standard-WSL2
- CPU: Intel i7-9750H (12) @ 2.592GHz

- gcc 11.1.0
- cmake 3.22.1

#### Python 部分

- Operating System: Windows 11 专业版 64-bit (10.0, Build 22000) (22000.co_release.210604-1628)
- Processor: Intel(R) Core(TM) i7-9750H CPU @ 2.60GHz (12 CPUs), ~2.6GHz
- Card name: NVIDIA GeForce GTX 1660 Ti

- Python 3.9.10

### 运行方式

#### C++ 部分

```sh
mkdir -p build
cd build
cmake ..
make -j
cd ..
./exp3 /path/to/DoubanMusic.txt
```

将生成 `douban.inter`，为 RecBole 所需数据集

#### Python 部分

```sh
git clone https://github.com/RUCAIBox/RecBole.git
mkdir -p RecBole/dataset/douban
cp ./main.py RecBole
cp ./predict.py RecBole
cp ./build/douban.inter RecBole/dataset/douban
cd RecBole
pip install -e .
```

将安装 RecBole 和所需依赖，并复制训练及预测脚本和实验所需数据集

执行 `python ./train.py` 进行训练，几分钟后输出最佳结果和模型文件
执行 `python ./predict.py /path/to/model.pth` 预测结果并输出

### 关键函数

#### C++ 部分

```cxx
constexpr size_t num_user = 23599;
constexpr size_t num_item = 21602;
constexpr size_t ratio = 20;

auto map = array<array<int, num_item>, num_user>{};
auto avg = array<int, num_item>{};
auto num = array<int, num_item>{};
```

```cxx
for (size_t iid = 0; iid != num_item; ++iid) {
    if (num[iid]) {
        avg[iid] /= num[iid];
    } else {
        avg[iid] = ratio;
    }
}
for (size_t uid = 0; uid != num_user; ++uid) {
    for (size_t iid = 0; iid != num_item; ++iid) {
        if (map[uid][iid] == -1) {
            map[uid][iid] = avg[iid];
        }
    }
}
```

使用矩阵存储所有输入，将评分放大至 `ratio` 倍（此处为 20，即范围为 20~100）以提高精度，使用每个 `item` 的平均评分替代 `-1`（我们认为一些人可能习惯于不评分，为其取评分的平均数不影响整体格局），若该 `item` 的所有评分均为 `-1` 则统一指定为 `ratio`（我们认为无人评分则质量不高）

#### Python 部分

> train.py

```python
config_dict = {
    # general
    'save_dataset': True,
    'gpu_id': 1,
    # evaluation settings
    'topk': [20, 100],
    'valid_metric': 'Hit@20',
    'eval_batch_size': 4096000,
}
```

使用 `HR@20` 作为评价选项进行迭代，扩大 `eval_batch_size` 以提高 `evaluate` 速度（训练时长仅几分钟）

> predict.py

```python
config, model, dataset, _, _, test_data = load_data_and_model(
    sys.argv[1])
user_list = [str(i) for i in range(23599)]
uids = dataset.token2id(dataset.uid_field, user_list)
_, iids = full_sort_topk(uids, model, test_data, 100, config['device'])
item_list = dataset.id2token(dataset.iid_field, iids.cpu())
```

读取模型文件中信息，预测前 100 项

## model_B：基于矩阵分解的推荐系统

### 环境

```
- Windows 10
- Intel(R) Core(TM) i7-9750H CPU @ 2.60GHz (12 CPUs), ~2.6GHz
- Python 3.10
```

### 文件结构

```
Matrix-Factorization
  ├─data
  │   ├── Basic_MF
  |   |     ├── itemMatrix.txt
  |   |     └── userMatrix
  │   ├── out
  |   |     ├── rating.txt
  |   |     └── out.txt
  |   ├── DoubanMusic.txt
  │   └── train.csv
  ├── csv.py
  └── main.py
```

### 运行方法

```shell
python ./csv.py
python ./main.py
```

### 关键函数

- `casv.py`
  - `preprocess` 函数预处理将源文件`DoubanMusic.txt`输出为 csv 文件`train.csv`，方便处理
- `main.py`
  - `matrix_factorization`函数将矩阵 R 分解成 P,Q
  - `getData`函数获取本地数据
  - `getUserItem`函数生成用户-物品矩阵并保存到本地文件中
  - `train`函数调用`matrix_factorization`函数进行训练
  - `topk`函数生成 topk 推荐列表
  - `test`函数读取用户矩阵，调用`topk`函数生成所有用户的推荐列表
  - `main`函数调用以上函数实现总体功能
