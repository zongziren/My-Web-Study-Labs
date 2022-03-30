import pandas as pd


def preprocess(input_file: str, output_path: str):
    df_list = []
    with open(input_file) as f:
        for line in f.readlines():
            first = True
            for token in line.split():
                if first:
                    user_id = token
                    first = False
                else:
                    item_id, rating = token.split(',')
                    df_list.append([user_id, item_id, rating])
    df = pd.DataFrame(df_list)
    df.to_csv(f'{output_path}/train.csv', ' ', index=False)


if __name__ == '__main__':
    preprocess("../data/DoubanMusic.txt", "../data")
