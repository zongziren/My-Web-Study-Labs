# exp 3 实验报告

- PB19000361 郑睿祺
- PB19000362 钟书锐

## 实验目的

根据豆瓣数据集，结合课程中推荐系统（第 14 节）、社会网络（第 15-16 节）的内容，查阅相关资料，合理设计模型，为每个用户提供音乐方面的个性化推荐（Personalized Recommendation），生成 Top-N 列表

## 实验步骤与核心思路

### model_A

#### 对数据集的分析及预处理

使用矩阵存储所有输入，将评分放大至 `ratio` 倍（此处为 20，即范围为 20~100）以提高精度，使用每个 `item` 的平均评分替代 `-1`（我们认为一些人可能习惯于不评分，为其取评分的平均数不影响整体格局），若该 `item` 的所有评分均为 `-1` 则统一指定为 `ratio`（我们认为无人评分则质量不高）

### 核心思路

使用 [RecBole](https://recbole.io/cn/index.html) 中的 BPR 模型进行来自隐式反馈的贝叶斯个性化排名

![RecBole Overall Architecture](https://raw.githubusercontent.com/RUCAIBox/RecBole/master/asset/framework.png)

### model_B：基于矩阵分解的推荐系统

### 对数据集的分析及预处理

首先通过 csv.py 将原始数据转换为 csv 格式数据方便处理

### 核心思路

确定损失函数和确定超参数惩罚系数。
利用交替最小二乘法分解稀疏的用户评分矩阵，得到用户因子矩阵和物品（电影）因子矩阵
梯度下降分解矩阵,所用参数如下，设置 alpha=0.05, Lambda=0.002，迭代运算

```
    R:用户-物品对应的共现矩阵 m*n
    P:用户因子矩阵 m*k
    Q:物品因子矩阵 k*n
    K:隐向量的维度
    steps:最大迭代次数
    alpha:学习率
    Lambda:L2正则化的权重系数
```

通过分解完成的矩阵求得每个用户对所有商品的评价，剔除掉已交互的商品，进行计算，返回前 100 的结果

## 测试与结果分析

### model_A

![result_model_A](submit/best_result_model_A.png)

与 Baseline 相差较小，实验结果较好

### model_B：基于矩阵分解的推荐系统

![result_model_B](submit/best_result_model_B.png)

与 Baseline 相差较大，实验结果不太好，主要原因迭代次数太少，只进行了三次迭代，拟合不够，但是因为单次迭代太耗费时间，所以只迭代了三次
