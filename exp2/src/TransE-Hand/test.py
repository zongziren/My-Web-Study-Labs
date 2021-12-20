import codecs
import random
import math

import numpy as np

entityId2vec = {}
relationId2vec = {}


def data_loader(file):
    file1 = file + "test.txt"
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

            triple_list.append([h_, 0,  r_])

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
    file1 = file + "entity_100dim1"
    file2 = file + "relation_100dim1"
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


def mean_rank(entity_set, triple_list, file):
    # triple_batch = random.sample(triple_list, 100)
    fileout = file + "out.txt"
    with codecs.open(fileout, "w") as f1:
        triple_batch = triple_list
        for triple in triple_batch:
            dlist = []
            h = triple[0]
            r = triple[2]

            for t_ in entity_set:
                if t_ in entityId2vec.keys() and r in relationId2vec.keys() and h in entityId2vec.keys():
                    dlist.append(
                        (t_, distance(entityId2vec[h], relationId2vec[r], entityId2vec[t_])))

            dlist = sorted(dlist, key=lambda val: val[1])
            if(len(dlist) > 1):
                f1.write(dlist[0][0])
                f1.write(',')
                f1.write(dlist[1][0])
                f1.write(',')
                f1.write(dlist[2][0])
                f1.write(',')
                f1.write(dlist[3][0])
                f1.write(',')
                f1.write(dlist[4][0])
                if(triple != triple_batch[-1]):
                    f1.write('\n')
            else:
                f1.write('0,0,0,0,0')
                if(triple != triple_batch[-1]):
                    f1.write('\n')


if __name__ == '__main__':
    file1 = ".\\data\\"
    file2 = ".\\out\\"
    print("load file...")
    entity_set, relation_set, triple_list = data_loader(file1)
    print("Complete load. entity : %d , relation : %d , triple : %d" % (
        len(entity_set), len(relation_set), len(triple_list)))
    print("load transE vec...")
    transE_loader(".\\")
    print("Complete load.")
    mean_rank(entity_set, triple_list, file2)
