import time
import string
import re
import os

import json
import urllib
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

import gensim
from gensim.models.doc2vec import Doc2Vec
# stop_text = open('stop_list.txt', 'r')
# stop_word = []
# for line in stop_text:
#     stop_word.append(line.strip())
TaggededDocument = gensim.models.doc2vec.TaggedDocument


def get_corpus(file_paths: list[str]):
    train_docs = []
    for i in file_paths:
        word_list = PreProcessing.get_tokens(File.get_text(i))
        document = TaggededDocument(word_list, tags=[i])
        train_docs.append(document)
    return train_docs


def train(x_train):

    model_dm = Doc2Vec(x_train, min_count=1, window=3,
                       vector_size=200, sample=1e-3, negative=5, workers=4)

    model_dm.train(x_train, total_examples=model_dm.corpus_count, epochs=70)
    model_dm.save('output/model_doc2vec')
    return model_dm


def test():

    model_dm = Doc2Vec.load("output/model_doc2vec")
    text_test = "Donald Trump declares trade war on china"
    text_cut = PreProcessing.get_tokens(text_test)
    #text_cut = jieba.cut(text_test)
    text_raw = []
    for i in list(text_cut):
        text_raw.append(i)
    inferred_vector_dm = model_dm.infer_vector(text_raw)
    sims = model_dm.docvecs.most_similar([inferred_vector_dm], topn=10)
    return sims


class File:
    def __init__(self, path: str):
        self.path: str = path

    def get_files(self) -> list[str]:
        return [os.path.join(parent, file)
                for parent, _, files in os.walk(self.path) for file in files]

    @staticmethod
    def get_text(path: str) -> str:
        with open(path, 'r', encoding='utf_8') as f:
            blog = json.load(f)
        return blog["text"].encode("ascii", errors="ignore").decode()

    @staticmethod
    def get_title(path: str) -> str:
        with open(path, 'r', encoding='utf_8') as f:
            blog = json.load(f)
        return blog["thread"]["title"].encode("ascii", errors="ignore").decode()

    @staticmethod
    def get_picture(path: str) -> str:
        with open(path, 'r', encoding='utf_8') as f:
            blog = json.load(f)
        return blog["thread"]["main_image"].encode("ascii", errors="ignore").decode()


class PreProcessing:
    __stop_words: list[str] = stopwords.words('english')+['']
    __lemmatizer = WordNetLemmatizer()

    @staticmethod
    def __tokenize(text: str) -> list[str]:
        return [token.strip(string.punctuation).lower()
                for token in re.split("[' \n]", text)]

    @classmethod
    def lemmatize(cls, text: str) -> str:
        return cls.__lemmatizer.lemmatize(text).lower()

    @classmethod
    def get_tokens(cls, text: str) -> list[str]:
        return [cls.lemmatize(i) for i in cls.__tokenize(text)
                if i not in cls.__stop_words]


def change_work_dir() -> None:
    current_path: str = os.path.dirname(os.path.realpath(__file__))
    project_path: str = os.path.dirname(current_path)
    os.chdir(project_path)


def main() -> None:
    change_work_dir()
    dataset_path: str = "dataset/US_Financial_News_Articles"
    output_path: str = "output"
    files = File(dataset_path)
    file_paths: list[str] = files.get_files()

    x_train = get_corpus(file_paths[0:1000])
    model_dm = train(x_train)

    sims = test()
    print(sims)
    i = 1
    for filename in sims:
        pic = File.get_picture(filename[0])
        if(pic != ''):
            urllib.request.urlretrieve(pic, f"output/picture2/{i}.jpg")
        i += 1


if __name__ == '__main__':
    start = time.process_time()
    main()
    end = time.process_time()
    print(end - start)
