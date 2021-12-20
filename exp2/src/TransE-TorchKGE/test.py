import numpy as np

entityId2vec = {}
relationId2vec = {}


def data_loader(dataset_path):
    file1 = f'{dataset_path}/test.txt'
    file2 = f'{dataset_path}/train.txt'

    entity_set = set()
    relation_set = set()
    triple_list = []

    with open(file1, 'r') as f:
        content = f.readlines()
        for line in content:
            triple = line.strip().split('\t')
            if len(triple) != 3:
                continue

            h_ = triple[0]
            r_ = triple[1]

            triple_list.append([h_, 0, r_])

            entity_set.add(h_)
            relation_set.add(r_)

    with open(file2, 'r') as f:
        content = f.readlines()
        for line in content:
            triple = line.strip().split('\t')
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
    file1 = f'{file}/ent_100dim'
    file2 = f'{file}/rel_100dim'
    with open(file1, 'r') as f:
        content = f.readlines()
        for line in content:
            line = line.strip().split('\t')
            entityId2vec[line[0]] = eval(line[1])
    with open(file2, 'r') as f:
        content = f.readlines()
        for line in content:
            line = line.strip().split('\t')
            relationId2vec[line[0]] = eval(line[1])


def distance(h, r, t):
    h = np.array(h)
    r = np.array(r)
    t = np.array(t)
    s = h + r - t
    return np.linalg.norm(s)


def mean_rank(entity_set, triple_list, file):
    with open(file, 'w') as f:
        triple_batch = triple_list
        for triple in triple_batch:
            dlist = []
            h = triple[0]
            r = triple[2]

            for t_ in entity_set:
                dlist.append((t_,
                              distance(entityId2vec[h], relationId2vec[r],
                                       entityId2vec[t_])))

            dlist = sorted(dlist, key=lambda val: val[1])
            for i in range(5):
                f.write(f'{dlist[i][0]},')
            f.write('\n')


def main():
    path = '.'
    output_path = f'{path}/out.txt'
    dataset_path = f'{path}/data'
    print('load file...')
    ent, rel, triple = data_loader(dataset_path)
    print(f'ent: {len(ent)} , rel: {len(rel)} , triple: {len(triple)}')
    print('load transE vec...')
    transE_loader('.')
    print('Complete load.')
    mean_rank(ent, triple, output_path)


if __name__ == '__main__':
    main()
