import json
import os
import re
import string
import time

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
    __stop_words: list[str] = stopwords.words('english') + ['']
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
    def __init__(self):
        self.file_dict: dict[int, str] = {}
        self.token_dict: dict[str, set[int]] = {}

    def build(self, file_paths: list[str]):
        for i, file_path in enumerate(file_paths):
            self.file_dict[i] = file_path

            text: str = File.get_text(file_path)
            tokens: list[str] = PreProcessing.get_tokens(text)
            for token in tokens:
                if token in self.token_dict.keys():
                    self.token_dict[token].add(i)
                else:
                    self.token_dict[token] = {i}

    def read(self, input_path: str):
        with open(f"{input_path}/invtered_index.json", 'r') as f:
            file: dict[str, list[int]] = json.load(f)
        self.token_dict = {i: set(file[i]) for i in file.keys()}
        with open(f"{input_path}/file_dict.json", 'r') as f:
            file: dict[str, str] = json.load(f)
        self.file_dict = {int(i): file[i] for i in file.keys()}

    def output(self, output_path: str):
        with open(f"{output_path}/invtered_index.json", 'w') as f:
            json.dump(self.token_dict, f,
                      default=lambda obj: list(obj)
                      if isinstance(obj, set) else obj)
        with open(f"{output_path}/file_dict.json", 'w') as f:
            json.dump(self.file_dict, f)

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

    def bool_search(self, text: str) -> None:
        text = text.replace('(', '( ').replace(')', ' )') \
            .replace('NOT', '-').replace('AND', '&').replace('OR', '|')
        tokens: list[str | set[int]] = []
        for token in text.split():
            if token in ['(', ')', '-', '&', '|']:
                tokens.append(token)
            elif PreProcessing.lemmatize(token) in self.token_dict.keys():
                tokens.append(self.token_dict[PreProcessing.lemmatize(token)])
            else:
                tokens.append(set())

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
        self.__print_result(self.__calculate(tokens))

    def __get_file_names(self, index: set[int]) -> list[str]:
        return [self.file_dict[i] for i in index]

    def __print_result(self, index: set[int]) -> None:
        for file in self.__get_file_names(index):
            print(f"{file}: {File.get_title(file)}")


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

    invtered_index = InvertedIndex()
    invtered_index.build(file_paths)
    invtered_index.output(output_path)
    invtered_index.read(output_path)
    search_text: str = input('input: ')
    # search_text: str = "Trump AND like AND China AND sale AND bank AND NOT market"
    invtered_index.bool_search(search_text)


if __name__ == '__main__':
    start = time.process_time()
    main()
    end = time.process_time()
    print(end - start)
