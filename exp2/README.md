# exp 2
- PB19000362 钟书锐
- PB19000361 郑睿祺

## TrabsE-Hand

- 手动完成（不依赖框架）的基于TrabsE算法的预测

#### 1. 环境依赖
- Windows 10
- CPU：Intel(R) Core(TM) i7-10750H CPU @ 2.60GHz
- Python 3.9.7
- Requirment:


#### 2. 文件介绍
The TransE-Hand includes the following files and directories:

├── data               
│   ├── dev.txt
│   ├── test.txt
│   ├── train.txt 
├── out
│   ├── out.txt 
├── transe.py     
├── dev.py
├── test.py

#### 3. 运行方法
1. 执行 `python.exe transe.py` 生成 entity_100dim1和relation_100dim1
2. 执行 `python.exe dev.py` 可以验证目前的hit
3. 执行 `python.exe test.py` 可以输出out.txt到 out文件夹下用来提交

## TrabsE-Hand-word2vec

- 在TrabsE-Hand的基础上利用word2vec使用文本信息来初始化

#### 1. 环境依赖
- Windows 10
- CPU：Intel(R) Core(TM) i7-10750H CPU @ 2.60GHz
- Python 3.9.7
- Requirment:

#### 2. 文件介绍
The TransE-Hand includes the following files and directories:

├── data               
│   ├── dev.txt
│   ├── test.txt
│   ├── train.txt 
├── out
│   ├── out.txt 
├── transe.py     
├── dev.py
├── test.py


#### 3. 运行方法
1. 执行 `python.exe entity.py` 生成entity_co.txt
2. 执行 `python.exe relation.py` 生成relation_co.txt
3. 执行 `python.exe transe.py` 生成 entity_100dim1和relation_100dim1
4. 执行 `python.exe dev.py` 可以验证目前的hit
5. 执行 `python.exe test.py` 可以输出out.txt到 out文件夹下用来提交