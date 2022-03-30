import numpy as np
import pandas as pd
import os
import time
import math
import codecs


'''
参数说明
    R:用户-物品对应的共现矩阵 m*n
    P:用户因子矩阵 m*k
    Q:物品因子矩阵 k*n
    K:隐向量的维度
    steps:最大迭代次数
    alpha:学习率
    Lambda:L2正则化的权重系数
'''


# 将矩阵R分解成P,Q
def matrix_factorization(R, P, Q, K, steps, alpha=0.5, Lambda=0.2):
    # 总时长
    sum_st = 0
    # 前一次的损失大小
    e_old = 0
    # 程序结束的标识
    flag = 1
    # 梯度下降结束条件1：满足最大迭代次数
    for step in range(steps):
        print("step!")
        # 每次跌代开始的时间
        st = time.time()
        cnt = 0
        e_new = 0
        for u in range(1, len(R)):
            for i in range(1, len(R[u])):
                if R[u][i] > 0:
                    eui = R[u][i] - np.dot(P[u, :], Q[:, i])
                    for k in range(K):
                        temp = P[u][k]
                        P[u][k] = P[u][k] + alpha * eui * \
                            Q[k][i] - Lambda * P[u][k]
                        Q[k][i] = Q[k][i] + alpha * \
                            eui * temp - Lambda * Q[k][i]
        for u in range(1, len(R)):
            for i in range(1, len(R[u])):
                if R[u][i] > 0:
                    cnt += 1
                    e_new = e_new + pow(R[u][i] - np.dot(P[u, :], Q[:, i]), 2)
        e_new = e_new / cnt
        et = time.time()
        sum_st = sum_st + (et - st)
        # 第一次迭代不执行前后损失之差
        if step == 0:
            e_old = e_new
            continue
        # 梯度下降结束条件2：loss过小，跳出
        if e_new < 1e-3:
            flag = 2
            break
        # 梯度下降结束条件3：前后loss之差过小，跳出
        if (e_old - e_new) < 1e-10:
            flag = 3
            break
        else:
            e_old = e_new
    print(f'--------Summary---------\nThe type of jump out:{flag}\nTotal steps:{step+1}\nTotal time:{sum_st}\n'
          f'Average time:{sum_st / (step+1)}\nThe e is :{e_new}')
    return P, Q


# 获取本地数据
def getData(path, dformat):
    # 读取训练集数据
    trainData = pd.read_csv(path+"/train.csv", sep=' ',
                            header=None, names=dformat)
    # 总用户数量
    all_user = np.unique(trainData['user'])
    # 总项目数量
    all_item = np.unique(trainData['item'])
    return trainData, all_user, all_item


# 生成用户-物品矩阵并保存到本地文件中
def getUserItem(path, train, all_user, all_item):
    # train.sort_values(by=['user, item'], axis=0, inplace=True)
    # 用户-项目共现矩阵行数
    num_user = len(all_user)+1
    print(num_user)
    # 用户-项目共现矩阵列数
    num_item = len(all_item)+1
    print(num_item)
    # 用户-项目共现矩阵初始化
    rating_mat = np.zeros([num_user, num_item], dtype=int)
    count = 0
    # 用户-项目共现矩阵赋值
    for i in range(len(train)):
        user = train.iloc[i]['user']
        item = train.iloc[i]['item']
        score = train.iloc[i]['score']
        if (score == -1):
            score = 0
        rating_mat[user][item] = score
        print(count)
        count += 1
    # 判断文件夹是否存在
    if os.path.exists(path+"/out"):
        pass
    else:
        os.mkdir(path+"/out")
    # 保存用户-项目共现矩阵到文件
    np.savetxt(path+"/out/rating.txt", rating_mat,
               fmt='%d', delimiter=' ', newline='\n')
    print(f'generate rating matrix complete!')
    return rating_mat


#	训练
def train(path, rating, K, steps):
    R = rating
    M = len(R)
    N = len(R[0])
    # 用户矩阵初始化
    P = np.random.normal(loc=0, scale=0.01, size=(M, K))
    print("P!")
    # 项目矩阵初始化
    Q = np.random.normal(loc=0, scale=0.01, size=(K, N))
    print("Q!")
    P, Q = matrix_factorization(R, P, Q, K, steps)
    # 判断文件夹是否存在
    if os.path.exists(path + "\\Basic_MF"):
        pass
    else:
        os.mkdir(path + "\\Basic_MF")
    # 将P，Q保存到文件
    np.savetxt(path+"\\Basic_MF\\userMatrix.txt", P,
               fmt="%.6f", delimiter=',', newline='\n')
    np.savetxt(path+"\\Basic_MF\\itemMatrix.txt", Q,
               fmt="%.6f", delimiter=',', newline='\n')
    print("train complete!")


# 生成topk推荐列表
def topK(dic, k):
    keys = []
    values = []
    for i in range(k):
        key, value = max(dic.items(), key=lambda x: x[1])
        keys.append(key)
        values.append(value)
        dic.pop(key)
    return keys, values

# 测试


def test(path, trainData, all_item, k):
    # 读取用户矩阵
    P = np.loadtxt(path+"\\Basic_MF\\userMatrix.txt",
                   delimiter=',', dtype=float)
    # 读取项目矩阵
    Q = np.loadtxt(path+"\\Basic_MF\\itemMatrix.txt",
                   delimiter=',', dtype=float)
    # 测试集中的用户集合
    testUser = np.unique(trainData['user'])

    fileout = path + "/out/out.txt"

    with codecs.open(fileout, "w") as f1:
        for user_i in testUser:
            # 测试集第i个用户在训练集已访问的项目
            visited_list = list(trainData[trainData['user'] == user_i]['item'])
            # 生成测试集第i个用户未访问的项目:评分对
            poss = {}
            for item in all_item:
                if item in visited_list:
                    continue
                else:
                    poss[item] = np.dot(P[user_i, :], Q[:, item])
            # 生成第i个用户的推荐列表
            ranked_list, test_score = topK(poss, k)
            f1.write(str(user_i))
            f1.write('\t')
            for i in range(len(ranked_list)):
                if (i != len(ranked_list)-1):
                    f1.write(str(ranked_list[i])+',')
                else:
                    f1.write(str(ranked_list[i]))
            f1.write('\n')


if __name__ == '__main__':
    rtnames = ['user', 'item', 'score']
    path = "../data"
    trainData, all_user, all_item = getData(path, rtnames)
    rating = getUserItem(path, trainData, all_user, all_item)
    print("rating read!")
    train(path, rating, 30, 10)
    print("train compelete!")
    test(path, trainData, all_item, 100)
