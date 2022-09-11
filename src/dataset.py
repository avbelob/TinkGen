import numpy as np
import nltk
from gensim.models.word2vec import Word2Vec
from torch.utils.data import DataLoader
import torch
import os
import re
from .config import *
from collections import Counter


def load_text(dir_data_path):
    text = ""

    if dir_data_path is None:
        return input()

    for file in os.listdir(dir_data_path):
        if file.endswith(".txt"):
            text += open(os.path.join(dir_data_path, file), errors="ignore").read() + "\n"

    return text


class Dataset(torch.utils.data.Dataset):
    def __init__(self, sequence_length=sequence_length,
                 dir_data_path=None, language='english', w2v_size=input_size):
        self.sequence_length = sequence_length
        self.text = load_text(dir_data_path)
        self.text = re.sub(r'[^\w\s]', '', self.text).lower()

        # Уменьшаем датасет, чтобы влезть во временные рамки
        self.text = self.text[:len(self.text) // 2]

        sentences = self.get_sentences(language)
        self.w2v_model = Word2Vec(sentences=sentences, vector_size=w2v_size, min_count=1, workers=8)

        self.tokens = [t for sentence in sentences for t in sentence]
        self.tokens_vectors = np.array([self.w2v_model.wv[t] for t in self.tokens])
        self.word_counts = self.get_words_count

    def get_sentences(self, language):
        sentences = []
        for sentence in nltk.tokenize.sent_tokenize(self.text, language=language):
            sentences.append(list(nltk.tokenize.word_tokenize(sentence, language=language)))
        return sentences

    def word2idx(self, word):
        return self.w2v_model.wv.key_to_index[word]

    def idx2word(self, idx):
        return self.w2v_model.wv.index_to_key[idx]

    def get_vocab_size(self):
        return len(self.w2v_model.wv.key_to_index)

    def get_words_count(self):
        word_counts = Counter(self.tokens)
        return np.array([word_counts[self.idx2word(i)] for i in range(self.get_vocab_size())])

    def __len__(self):
        return len(self.tokens_vectors) - self.sequence_length

    def __getitem__(self, index):
        return (
            torch.tensor(self.tokens_vectors[index:index + self.sequence_length]),
            torch.tensor(self.tokens_vectors[index + 1:index + self.sequence_length + 1]),
        )
