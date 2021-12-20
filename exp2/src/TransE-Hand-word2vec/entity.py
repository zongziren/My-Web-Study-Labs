import gensim
import random
import math

import numpy as np

from gensim.models import word2vec
with open("entity_with_text.txt", 'r') as fr, open("entity_co.txt", 'w') as fw:
    for line in fr.readlines():
        fw.write(line.strip().split('\t')[1])
        fw.write("\n")
