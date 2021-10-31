# exp 1 实验报告

- PB19000361 郑睿祺

- PB19000362 钟书锐

## 一. 实验目的

以给定的财经新闻数据集为基础，实现一个新闻搜索引擎。对于给定的查询，能够以精确查询或模糊语义匹配的方式返回最相关的一系列新闻文档。

## 二. 实验环境

- 倒排索引 + 布尔查询使用
- Windows 11
- CPU：Intel(R) Core(TM) i7-9750H CPU @ 2.60GHz
- TF-IDF + 语义查询使用
- Windows 10
- CPU：Intel(R) Core(TM) i7-10750H CPU @ 2.60GHz
- Python 3.10.0
- Requirment:

```
gensim==4.1.2
nltk==3.6.5
numpy==1.21.3
scikit-learn==1.0.1
scipy==1.7.1
urllib3==1.26.7
```

## 二. 实验步骤与核心思路

### 总体框架

`File` 类执行对数据集的 I/O 操作
`PreProcessing` 类对输入的文本进行分词、去停用词、词根化等预处理
`InvertedIndex` 类处理倒排索引相关

### 预处理部分

1. 将新闻文本使用 `ascii` 编码以去除非 `ascii` 文本

1. `[' \n]` 作为分隔符对文本进行分割，再去除各词首尾的标点符号，得到分词化的文本

1. `nltk.corpus.stopwords` 中的 `'english'` 词库和 `['']` 作为停用词库，去停用词

1. 使用 `nltk.stem.WordNetLemmatizer` 进行词根化

### 倒排索引构建

使用 `file_dict: dict[int, str]` 存储文档编号

使用 `token_dict: dict[str, set[int]]` 存储倒排索引

遍历新闻，将所有 `token` 加入倒排索引中

```python
def build(self, file_paths: list[str]):
    for i, file_path in enumerate(file_paths):
        self.file_dict[i] = file_path
        text: str = File.get_text(file_path)
        tokens: list[str] = PreProcessing.get_tokens(text)
        for token in tokens:
            if token in self.token_dict.keys():
                self.token_dict[token].add(i)
            else:
                self.token_dict[token] = {i}
```

### 布尔查询

`AND, OR, NOT, (, )` 作为关键字，对输入的其他单词进行词根化处理

通过 Python 中 `set` 类型的交集、并集、差集运算完成对布尔表达式中 `AND, OR, NOT` 的操作
### TF-IDF 矩阵的计算

- 计算主要调用 sklearn 库
- Scikit-learn 是一个用于数据挖掘和数据分析的简单且有效的工具，它是基于 Python 的机器学习模块，基于 BSD 开源许可证。
- Scikit-learn 的基本功能主要被分为六个部分：分类(Classification)、回归(Regression)、聚类(Clustering)、数据降维(Dimensionality reduction)、模型选择(Model selection)、数据预处理(Preprocessing)。
- Scikit-Learn 中 TF-IDF 权重计算方法主要用到两个类：CountVectorizer 和 TfidfTransformer。
- CountVectorizer 类会将文本中的词语转换为词频矩阵。
- TfidfTransformer 用于统计 vectorizer 中每个词语的 TF-IDF 值。
- 核心代码如下

  ```
  self.vectorizer = CountVectorizer()
  self.transformer = TfidfTransformer()
  # 该类会统计每个词语的 tf-idf 权值
  self.tfidf = self.transformer.fit_transform(self.vectorizer.fit_transform(self.corpus))
  # 第一个 fit_transform 是计算 tf-idf，第二个 fit_transform 是将文本转为词频矩阵
  ```

  ```
  sparse.save_npz('output/b_compressed.npz', self.Atfidf, True)
  sparse.save_npz('output/b.npz', self.Atfidf, False)
  ```

  分别输出压缩后的矩阵和未压缩的 tfidf 矩阵

### 语义查询

- main 函数接受输入的 str，进行预处理（分词与去停用词）
- 使用类似 TF-IDF 矩阵的构造方法，为输入的字符串构造 tfidf 向量
- 查询向量与每个文档的 tfidf 向量相乘，结果存入 list
- 对 list 排序返回最相关的十个文件名

### TFIDFI 时空复杂度优化与对比

- 时间复杂度：
- 在最初的版本中我们使用 gensim 库来计算 tfidf 矩阵
- Gensim 是在做自然语言处理时较为经常用到的一个工具库，主要用来以无监督的方式从原始的非结构化文本当中来学习到文本隐藏层的主题向量表达。
- 主要包括 TF-IDF，LSA，LDA，word2vec，doc2vec 等多种模型。
- 我们对通过 Gensim 和 Scikit-Learn 2 种方法计算 tfidf 的矩阵的时间进行了对比，发现在 10000 个文档的情况 Scikit-Learn 比 Gensim 快数倍。
- 最终我们采用 Scikit-Learn 计算 tfidf 矩阵
- 同时输出 tfidf 矩阵，可以调用`sparse.save_npz()`，此时可以选择压缩，`sparse.save_npz('output/b_compressed.npz', self.Atfidf, True)`，减少占用空间。

### 附加：索引表的时空复杂度

起初倒排索引使用 `dict[str, list[int]]` 类型实现，对每篇文章中的每个 `token`，都需要查询 是否位于 `list[int]` 中，而 `in` 操作的平均时间复杂度为 $O(n)$，故对于 $m$ 篇文章和共 $n$ 个 `token` 而言，平均时间复杂度为 $O(mn)$

后改为 `dict[str, set[int]]`，不需要判重操作，平均时间复杂度降低为 $O(n)$

### 附加：doc2vec

- Doc2vec 方法是一种无监督算法，能从变长的文本（例如：句子、段落或文档）中学习得到固定长度的特征表示。Doc2vec 也可以叫做 Paragraph Vector、Sentence Embeddings，它可以获得句子、段落和文档的向量表达，是 Word2Vec 的拓展，其具有一些优点，比如不用固定句子长度，接受不同长度的句子做训练样本。Doc2vec 算法用于预测一个向量来表示不同的文档 ，该模型的结构潜在的克服了词袋模型的缺点。

- 使用 gensim 包进行训练，建立 models

- 我们使用 gensim.models.doc2vec.TaggedDocument 将预处理后的文档合并为函数可以接受的 train_docs

- 使用 `model_dm.train(x_train, total_examples=model_dm.corpus_count, epochs=70)` 语句进行训练并保存到本地

- 调用 `model_dm.docvecs.most_similar([inferred_vector_dm], topn=10)` 返回搜索结果最相关的十个文件的文件名

  ```
  def test():
    model_dm = Doc2Vec.load("model_doc2vec")
    text_test = u"Donald John Trump like china"
    text_cut = jieba.cut(text_test)
    text_raw = []
    for i in list(text_cut):
        text_raw.append(i)
    inferred_vector_dm = model_dm.infer_vector(text_raw)
    sims = model_dm.docvecs.most_similar([inferred_vector_dm], topn=10)
    return sims
  ```

- doc2vec 和 tfidf 结果对比将在测试与结果分析部分进行讨论

### 附加：语义查询返回图片

- 在语义查询结束后返回了最相关的十个文件的文件名
- 通过 get_picture 取得文件内的 main_image url 信息

```
@staticmethod
    def get_picture(path: str) -> str:
        with open(path, 'r', encoding='utf_8') as f:
            blog = json.load(f)
        return blog["thread"]["main_image"].encode("ascii", errors="ignore").decode()

```

- 对每个链接调用 urllib.request.urlretrieve，将对应的文件下载在{output_path}/picture 文件夹下

```
pic = File.get_picture(filename)
if(pic != ''):
    urllib.request.urlretrieve(pic, f"{output_path}/picture/{i}.jpg")
```

## 四. 测试与结果分析

### 倒排索引的建立

建立倒排索引耗时 `695.78125s`


### 布尔查询

对 `"Trump AND like AND China AND sale AND bank AND NOT market"` 进行查询，输出文章路径和文章标题

```
dataset/US_Financial_News_Articles\2018_01\blogs_0054317.json: Change the World, McCabe Out, Oligarch List: CEO Daily for January 30, 2018
dataset/US_Financial_News_Articles\2018_05\news_0054894.json: No illusions as Iran nuclear deal countries look to future without U.S
dataset/US_Financial_News_Articles\2018_05\news_0036130.json: Brazil's wealthy farm belt backs Trump-like presidential candidate
dataset/US_Financial_News_Articles\2018_05\blogs_0025616.json: AI, ZTE, RBS and BT: CEO Daily for May 10, 2018
dataset/US_Financial_News_Articles\2018_05\blogs_0042426.json: ANALYSIS-Brazil's wealthy farm belt backs Trump-like presidential candidate
dataset/US_Financial_News_Articles\2018_01\news_0021444.json: has kept many promises during his first year in office
dataset/US_Financial_News_Articles\2018_05\blogs_0030761.json: Stocks set to rally at the open, giving Dow a shot at extending 7-day win streak
dataset/US_Financial_News_Articles\2018_04\news_0036022.json: UPDATE 2-German investor morale tumbles as trade war worries grow
dataset/US_Financial_News_Articles\2018_05\news_0048644.json: ZTE Pushback, Dodd-Frank, Barclays Plans: CEO Daily for May 23, 2018 | Fortune
dataset/US_Financial_News_Articles\2018_04\news_0016342.json: Facebook's Mark Zuckerberg Goes to Congress to Talk Privacy | Fortune
dataset/US_Financial_News_Articles\2018_04\blogs_0061501.json: The Morning Risk Report: Huawei Looks to Avoid ZTEs Fate
dataset/US_Financial_News_Articles\2018_05\news_0028934.json: The Morning Risk Report: DOJ Takes Aim at Enforcement Piling On - Risk & Compliance Journal. - WSJ
dataset/US_Financial_News_Articles\2018_03\news_0031298.json: U.S. tech companies win changes in bill to limit China access to technology
dataset/US_Financial_News_Articles\2018_05\news_0034018.json: Brazil's wealthy farm belt backs Trump-like presidential candidate
dataset/US_Financial_News_Articles\2018_05\news_0054745.json: No illusions as Iran nuclear deal countries look to future without U.S
dataset/US_Financial_News_Articles\2018_04\news_0033243.json: German investor morale tumbles as trade war worries grow
dataset/US_Financial_News_Articles\2018_03\news_0032902.json: U.S. tech companies win changes in bill to limit China access to technology
dataset/US_Financial_News_Articles\2018_05\news_0054248.json: No illusions as Iran nuclear deal countries look to future without U.S.


```

耗时 `20.640625s`

其中读取倒排索引文件耗时 `19.78125s`

经检验，搜索结果符合预期

### tfidf 矩阵的建立

- 测试集合为[0:10000]即前 10000 个文件时，建立 tfidf 矩阵耗时 16.671875s
- 测试集合为[0:100000]即前 100000 个文件时，建立 tfidf 矩阵耗时 184.765625
- 测试集合为全部文件（大约 300000）时，建立 tfidf 矩阵耗时 716.921875s
- 上述时间不包含输出矩阵为文件的时间（IO 时间），测试中如果加上输出到本地 10000 个测试需要 18.421875s
- 对于输出的矩阵可以选择是否进行压缩，对于 10000 个测试集
- 输出的压缩后的 tfidf 矩阵为 12752kb
- 输出的未压缩的 tfidf 矩阵为 17266kb

### 语义查询

- 对于"Donald John Trump like china"进行查询，tfidf 测试集合为[0:100000]
- 输出

```
['dataset/US_Financial_News_Articles\\2018_01\\blogs_0032983.json',
'dataset/US_Financial_News_Articles\\2018_01\\news_0006043.json',
'dataset/US_Financial_News_Articles\\2018_01\\news_0002239.json',
'dataset/US_Financial_News_Articles\\2018_01\\blogs_0057533.json', '
dataset/US_Financial_News_Articles\\2018_01\\blogs_0041404.json',
'dataset/US_Financial_News_Articles\\2018_01\\blogs_0003894.json',
'dataset/US_Financial_News_Articles\\2018_01\\blogs_0051445.json',
'dataset/US_Financial_News_Articles\\2018_01\\blogs_0043096.json',
'dataset/US_Financial_News_Articles\\2018_01\\blogs_0045365.json',
'dataset/US_Financial_News_Articles\\2018_01\\blogs_0046260.json']
```

- 返回的第一篇新闻查阅后发现为"Trump is 'determined to bite somebody, and China is the most likely target,' trade expert says"查看内容后基本符合期望
- 下载到的对应的图片为,基本符合预期
  ![](./assest/1.jpg)
- 返回的第二篇新闻查阅后发现为"US-China 'trade war' could explode this year"查看内容后基本符合期望
- 下载到的对应的图片为,基本符合预期
  ![](./assest/2.jpg)

### tfidf 与 doc2vec 对比

- TF-IDF 这种方法没有考虑到文字背后的语义关联，可能在两个文档共同出现的单词很少甚至没有相同的单词，但两个文档是相似的情况下，就需要考虑到文档的语义。

- 输出

```
[('dataset/US_Financial_News_Articles\\2018_01\\blogs_0018149.json', 0.8506518006324768),
('dataset/US_Financial_News_Articles\\2018_01\\blogs_0005433.json', 0.8450565934181213),
('dataset/US_Financial_News_Articles\\2018_01\\blogs_0023376.json', 0.833734929561615),
('dataset/US_Financial_News_Articles\\2018_01\\blogs_0008031.json', 0.821879506111145),
('dataset/US_Financial_News_Articles\\2018_01\\blogs_0020739.json', 0.8164512515068054),
('dataset/US_Financial_News_Articles\\2018_01\\blogs_0020035.json', 0.8103011846542358),
('dataset/US_Financial_News_Articles\\2018_01\\blogs_0016890.json', 0.7895230054855347),
('dataset/US_Financial_News_Articles\\2018_01\\blogs_0013582.json', 0.7834425568580627),
('dataset/US_Financial_News_Articles\\2018_01\\blogs_0009534.json', 0.7810621857643127),
('dataset/US_Financial_News_Articles\\2018_01\\blogs_0016393.json', 0.7796798348426819)]
```

## 五.结论与实验总结
