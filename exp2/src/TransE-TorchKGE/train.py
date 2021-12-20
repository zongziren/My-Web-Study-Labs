from torch.optim import Adam
from torchkge.data_structures import KnowledgeGraph
from torchkge.models import TransEModel
from torchkge.utils import Trainer, MarginLoss

import numpy as np
import pandas as pd


def load_dataset(path):
    df_train = pd.read_csv(
        f'{path}/train.txt',
        sep='\t',
        header=None,
        names=['from', 'rel', 'to'],
        engine='c',
        dtype={
            'from': np.int16,
            'rel': np.int16,
            'to': np.int16,
        },
    )
    df_dev = pd.read_csv(
        f'{path}/dev.txt',
        sep='\t',
        header=None,
        names=['from', 'rel', 'to'],
        engine='c',
        dtype={
            'from': np.int16,
            'rel': np.int16,
            'to': np.int16,
        },
    )
    df_test = pd.read_csv(
        f'{path}/test.txt',
        sep='\t',
        header=None,
        names=['from', 'rel', 'to'],
        engine='c',
        dtype={
            'from': np.int16,
            'rel': np.int16,
            'to': np.int16,
        },
    )
    df = pd.concat([df_train, df_dev, df_test])
    kg = KnowledgeGraph(df)

    return kg.split_kg(sizes=(len(df_train), len(df_dev), len(df_test)))


def main():
    # Define some hyper-parameters for training
    emb_dim = 100
    lr = 0.0004
    margin = 0.5
    n_epochs = 1000
    batch_size = 32768

    # Load dataset
    kg_train, kg_dev, kg_test = load_dataset('./data')

    # Define the model and criterion
    model = TransEModel(emb_dim,
                        kg_train.n_ent,
                        kg_train.n_rel,
                        dissimilarity_type='L2')
    criterion = MarginLoss(margin)
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    trainer = Trainer(
        model,
        criterion,
        kg_train,
        n_epochs,
        batch_size,
        optimizer=optimizer,
        sampling_type='bern',
        use_cuda='all',
    )

    trainer.run()

    ent, rel = model.get_embeddings()
    ent_list, rel_list = ent.tolist(), rel.tolist()
    ent2ix, rel2ix = kg_train.ent2ix, kg_train.rel2ix
    with open(f'ent_{emb_dim}dim', 'w') as f:
        for k in ent2ix.keys():
            f.write(f'{k}\t{str(ent_list[ent2ix[k]])}\n')
    with open(f'rel_{emb_dim}dim', 'w') as f:
        for k in rel2ix.keys():
            f.write(f'{k}\t{str(rel_list[rel2ix[k]])}\n')


if __name__ == '__main__':
    main()