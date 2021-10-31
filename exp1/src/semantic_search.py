import json
import os
import re
import string
import time
import math

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer

import numpy as np
from scipy import sparse
import urllib

import gensim


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


class Tfidf:
    def __init__(self, file_paths: list[str]):
        self.file_dict: dict[int, str] = {}
        for i, file_path in enumerate(file_paths):
            self.file_dict[i] = file_path

        self.corpus: list[str] = [
            " ".join(PreProcessing.get_tokens(File.get_text(i)))
            for i in file_paths]

        self.vectorizer = CountVectorizer()
        self.transformer = TfidfTransformer()
        # 该类会统计每个词语的tf-idf权值
        # 第一个fit_transform是计算tf-idf，第二个fit_transform是将文本转为词频矩阵
        self.tfidf = self.transformer.fit_transform(
            self.vectorizer.fit_transform(self.corpus))
        self.word = self.vectorizer.get_feature_names()
        # 获取词袋模型中的所有词语
        # 将tf-idf矩阵抽取出来，元素a[i][j]表示j词在i类文本中的tf-idf权重
        self.Atfidf = sparse.csr_matrix(self.tfidf)
        self.diction = self.Atfidf.todok()

        f = open('output/word.txt', 'w')
        print(self.word, file=f)
        f.close()

        f = open('output/file_path.txt', 'w')
        print(file_paths, file=f)
        f.close()

        sparse.save_npz('output/b_compressed.npz', self.Atfidf, True)
        sparse.save_npz('output/b.npz', self.Atfidf, False)

    def semantic_search(self, text: str) -> list[str]:
        corpus: list[str] = [
            " ".join(PreProcessing.get_tokens(text))]
        vectorizer = CountVectorizer()
        transformer = TfidfTransformer()
        tfidf = transformer.fit_transform(vectorizer.fit_transform(corpus))
        words = vectorizer.get_feature_names()
        weight = tfidf.toarray()
        simil: list[float] = []
        for i in range(self.tfidf.shape[0]):
            simil.append(0)
            for j in range(len(words)):
                word = words[j]
                if(word in self.word):
                    word_index = self.word.index(word)
                    value = self.diction.get((i, word_index))
                    if value:
                        simil[i] += value*weight[0][j]
        simi_sort = np.array(simil)
        out_sort = np.argsort(-simi_sort)
        out_list = out_sort.tolist()
        out = out_list[0:10]
        return self.__get_file_names(out)

    def __get_file_names(self, index: set[int]) -> list[str]:
        return [self.file_dict[i] for i in index]


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
    tfidf = Tfidf(file_paths[0:10000])
    search_text: str = "Donald John Trump declares trade war on china"

    s = tfidf.semantic_search(search_text)
    print(s)

    i = 1
    for filename in s:
        pic = File.get_picture(filename)
        if(pic != ''):
            urllib.request.urlretrieve(pic, f"{output_path}/picture/{i}.jpg")
        i += 1

    '''
    search_text: str = input('input: ')
    while(search_text):
        s = tfidf.semantic_search(search_text)
        print(s)
        for filename in s:
            print(File.get_picture(filename))
        search_text = input('input: ')
    '''


if __name__ == '__main__':
    start = time.process_time()
    main()
    end = time.process_time()
    print(end - start)
