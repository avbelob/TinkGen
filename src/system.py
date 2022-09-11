import torch
from torch.utils.data import DataLoader
from torch import nn, optim
from .model import LSTMModel
from .dataset import Dataset
from .config import *
from tqdm import trange, tqdm
import nltk
import numpy as np
import string


class TextGeneratorSystem:
    def __init__(self):
        self.dataset = None
        self.model = None
        self.optimizer = None
        self.criterion = nn.MSELoss()
        nltk.download("punkt", quiet=True)

    def fit(self, input_dir=None, model_path=None):
        self.dataset = Dataset(sequence_length=sequence_length, dir_data_path=input_dir)

        self.model = LSTMModel()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        if model_path is not None:
            checkpoint = torch.load(model_path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        dataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True)

        for _ in tqdm(range(max_epochs)):
            losses = []
            state_h, state_c = self.model.init_state(batch_size=batch_size)

            iter_number = len(self.dataset) // batch_size
            pbar = trange(iter_number)

            for _, (x, y) in zip(pbar, dataloader):
                self.optimizer.zero_grad()
                y_pred, (state_h, state_c) = self.model(x, (state_h, state_c))

                loss = self.criterion(y_pred, y)
                losses.append(loss.cpu().detach().numpy())
                pbar.set_postfix({'loss': '{:.5f}'.format(losses[-1])})
                state_h = state_h.detach()
                state_c = state_c.detach()

                loss.backward()

                self.optimizer.step()

            print()
            print("Mean loss per epoch: ", np.mean(losses))

        print("Обучение завершено!")

        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'input_dir': input_dir,
        }, "model.pkl")

        print("Модель сохранена!")

    def generate(self, model_path, length, prefix=None):
        checkpoint = torch.load(model_path)
        self.model = LSTMModel()
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.dataset = Dataset(sequence_length=sequence_length, dir_data_path=checkpoint['input_dir'])

        if len(prefix) == 0:
            vocab = list(self.dataset.w2v_model.wv.key_to_index.keys())
            prefix = vocab[np.random.randint(len(vocab))]

        words = nltk.tokenize.word_tokenize(prefix.lower())

        self.model.eval()

        state_h, state_c = self.model.init_state()

        for i in range(length):
            x = torch.tensor(np.array([[self.dataset.w2v_model.wv[w] for w in words[i:]]]))
            y_pred, (state_h, state_c) = self.model(x, (state_h, state_c))

            next_word = self.dataset.w2v_model.wv.similar_by_vector(y_pred[0][-1].detach().numpy())[0]
            if next_word[0] == words[-1]:
                top_words = self.dataset.w2v_model.wv.similar_by_vector(y_pred[0][-1].detach().numpy(), topn=6)[1:]
                next_word = top_words[np.random.randint(len(top_words))]
            words.append(next_word[0])

        return "".join([" " + i if not i.startswith("'") and i not in string.punctuation else i for i in words]).strip()
