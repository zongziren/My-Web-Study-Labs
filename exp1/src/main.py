import json
import math
import os
import re
import string
import time
from collections import Counter

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


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


class PreProcessing:
    __stop_words: list[str] = stopwords.words('english')
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


class InvertedIndex:
    def __init__(self, file_paths: list[str]):
        self.file_dict: dict[int, str] = {}
        self.token_dict: dict[str, list[int]] = {}
        self.dft: dict[str, int] = {}
        self.idft: dict[str, float] = {}
        self.tf: dict[int, Counter[str]] = {}
        self.tfidft: dict[int, Counter[str]] = {}
        N = 0

        for i, file_path in enumerate(file_paths):
            self.file_dict[i] = file_path

            blog_text = File.get_text(file_path)
            tokens: list[str] = PreProcessing.get_tokens(blog_text)

            for token in tokens:
                if token in self.token_dict.keys():
                    if i not in self.token_dict[token]:
                        self.token_dict[token].append(i)
                        self.dft[token] += 1
                else:
                    self.token_dict[token] = [i]
                    self.dft[token] = 1

            count = Counter(tokens)
            self.tf[i] = count
            N += 1

        for token in self.dft:
            self.idft[token] = math.log10(N / self.dft[token])

        for index in self.tf:
            self.tfidft[index] = self.tf[index]
            for token in self.tf[index]:
                tfnum = self.tf[index][token]
                wd = (1 + math.log10(tfnum)) * self.idft[token]
                self.tfidft[index][token] = wd

    def output(self, output_path: str):
        with open(output_path, 'w', encoding='utf_8') as f:
            json.dump(self.token_dict, f)

    def output_tfidft(self, output_path: str):
        with open(output_path, 'w', encoding='utf_8') as f:
            json.dump(self.tfidft, f)

    def __calculate(self, tokens: list[str | set[int]]) -> set[int]:
        all_token: set[int] = {i for i in range(len(self.file_dict))}
        stack: list[set[int]] = []
        i = 0
        while True:
            if i == len(tokens):
                break
            if tokens[i] == '-':
                stack.append(all_token - tokens[i + 1])
                i += 2
            elif tokens[i] in ['&', '|']:
                set1: set[int] = stack.pop()
                if tokens[i + 1] == '-':
                    set2: set[int] = all_token - tokens[i + 2]
                else:
                    set2: set[int] = tokens[i + 1]
                if tokens[i] == '&':
                    stack.append(set1 & set2)
                else:
                    stack.append(set1 | set2)
                if tokens[i + 1] == '-':
                    i += 3
                else:
                    i += 2
            else:
                stack.append(tokens[i])
                i += 1
        return stack.pop()

    def __search(self, token: str) -> set[int]:
        if token in self.token_dict.keys():
            return set(self.token_dict[token])
        else:
            return set()

    def bool_search(self, text: str) -> list[str]:
        text = text.replace('(', '( ').replace(')', ' )') \
            .replace('NOT', '-').replace('AND', '&').replace('OR', '|')
        tokens: list[str | set[int]] = [
            token if token in ['(', ')', '-', '&', '|']
            else self.__search(PreProcessing.lemmatize(token))
            for token in text.split()]
        while '(' in tokens:
            right: int = tokens.index(')')
            left: int = 0
            for i, token in enumerate(reversed(tokens)):
                if token == '(':
                    left = len(tokens) - i - 1
            if right == len(tokens):
                tokens = tokens[:left] + \
                         [self.__calculate(tokens[left + 1:right])]
            else:
                tokens = tokens[:left] + \
                         [self.__calculate(tokens[left + 1:right])] + \
                         tokens[right + 1:]
        result: set[int] = self.__calculate(tokens)

        return self.__get_file_names(result)

    def semantic_search(self, text: str) -> list[str]:
        tokens: list[str] = PreProcessing.get_tokens(text)
        return self.__get_file_names({0})

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

    invtered_index = InvertedIndex(file_paths[0:10001])
    invtered_index.output(f"{output_path}/ii.json")
    invtered_index.output_tfidft(f"{output_path}/tfidft.json")
    # search_text: str = input('input: ')
    search_text: str = "sale AND NOT tax AND (NOT loss) AND revenue"
    result: list[str] = invtered_index.bool_search(search_text)
    print(result)


if __name__ == '__main__':
    start = time.process_time()
    main()
    end = time.process_time()
    print(end - start)
