import codecs
import random
import math

import numpy as np

entityId2vec = {}
relationId2vec = {}


def data_loader(file):
    file1 = file + "dev.txt"
    file2 = file + "train.txt"

    entity_set = set()
    relation_set = set()
    triple_list = []

    with codecs.open(file1, 'r') as f:
        content = f.readlines()
        for line in content:
            triple = line.strip().split("\t")
            if len(triple) != 3:
                continue

            h_ = triple[0]
            r_ = triple[1]
            t_ = triple[2]

            triple_list.append([h_, t_,  r_])

            entity_set.add(h_)
            relation_set.add(r_)

    with codecs.open(file2, 'r') as f:
        content = f.readlines()
        for line in content:
            triple = line.strip().split("\t")
            if len(triple) != 3:
                continue

            h_ = triple[0]
            r_ = triple[1]
            t_ = triple[2]

            entity_set.add(h_)
            entity_set.add(t_)
            relation_set.add(r_)

    return entity_set, relation_set, triple_list


def transE_loader(file):
    file1 = file + "entity_50dim1"
    # file1 = file + "entity_50dim"
    file2 = file + "relation_50dim1"
    # file2 = file + "relation_50dim"
    with codecs.open(file1, 'r') as f:
        content = f.readlines()
        for line in content:
            line = line.strip().split("\t")
            entityId2vec[line[0]] = eval(line[1])
    with codecs.open(file2, 'r') as f:
        content = f.readlines()
        for line in content:
            line = line.strip().split("\t")
            relationId2vec[line[0]] = eval(line[1])


def distance(h, r, t):
    h = np.array(h)
    r = np.array(r)
    t = np.array(t)
    s = h + r - t
    return np.linalg.norm(s)


def mean_rank(entity_set, triple_list):
    # triple_batch = random.sample(triple_list, 100)
    triple_batch = triple_list
    mean = 0
    hit5 = 0
    hit1 = 0
    for triple in triple_batch:
        dlist = []
        h = triple[0]
        t = triple[1]
        r = triple[2]
        if t in entityId2vec.keys():
            if r in relationId2vec.keys():
                if h in entityId2vec.keys():
                    dlist.append(
                        (t, distance(entityId2vec[h], relationId2vec[r], entityId2vec[t])))
        for t_ in entity_set:
            if t_ != t:
                if t_ in entityId2vec.keys():
                    if r in relationId2vec.keys():
                        if h in entityId2vec.keys():
                            dlist.append(
                                (t_, distance(entityId2vec[h], relationId2vec[r], entityId2vec[t_])))
        dlist = sorted(dlist, key=lambda val: val[1])
        for index in range(len(dlist)):
            if dlist[index][0] == t:
                mean += index + 1
                if index < 1:
                    hit1 += 1
                if index < 5:
                    hit5 += 1
                # print(index)
                break
    print("mean rank:", mean / len(triple_batch))
    print("hit@1:", hit1 / len(triple_batch))
    print("hit@5:", hit5 / len(triple_batch))


if __name__ == '__main__':
    file1 = ".\\data\\"
    print("load file...")
    entity_set, relation_set, triple_list = data_loader(file1)
    print("Complete load. entity : %d , relation : %d , triple : %d" % (
        len(entity_set), len(relation_set), len(triple_list)))
    print("load transE vec...")
    transE_loader(".\\")
    print("Complete load.")
    mean_rank(entity_set, triple_list)
