import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from config import config


class Word_CNN(nn.Module):
    def __init__(self, dataset, hidden_dims=250):
        super(Word_CNN, self).__init__()
        max_len = config.word_max_len[dataset]
        num_classes = config.num_classes[dataset]
        loss = config.loss[dataset]
        kernel_size = 3

        embedding_dims = config.wordCNN_embedding_dims[dataset]
        num_words = config.num_words[dataset]
        self.dataset = dataset
        self.embedding = nn.Embedding(num_words, embedding_dims)
        self.conv1d = nn.Conv1d(in_channels=embedding_dims, out_channels=hidden_dims, kernel_size=kernel_size)
        self.linear1 = nn.Linear(in_features=hidden_dims, out_features=hidden_dims)
        self.linear2 = nn.Linear(in_features=hidden_dims, out_features=num_classes)

    def forward(self, x):
        x = self.embedding(x)
        x = x.view(x.size(0), 50, -1)
        x = self.conv1d(x)
        x = F.max_pool1d(x, kernel_size=x.size(-1))
        x = x.view(x.size(0), -1)
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x


class BDLSTM(nn.Module):
    def __init__(self, dataset):
        super(BDLSTM, self).__init__()
        max_len = config.word_max_len[dataset]
        num_classes = config.num_classes[dataset]
        loss = config.loss[dataset]
        embedding_dims = config.wordCNN_embedding_dims[dataset]
        num_words = config.num_words[dataset]
        self.dataset = dataset

        self.embedding = nn.Embedding(num_words, embedding_dims)
        self.lstm = nn.LSTM(embedding_dims, 64, bidirectional=True)
        self.dropout = nn.Dropout(0.5)
        self.linear = nn.Linear(128, num_classes)

    def forward(self, x):
        activation = config.activation[self.dataset]
        x = self.embedding(x)
        x, _ = self.lstm(x)
        x = self.dropout(x)
        x = self.linear(x)
        if activation == 'sigmoid':
            out = F.sigmoid(x)
        elif activation == 'softmax':
            out = F.softmax(x, dim=1)
        else:
            raise ValueError
        return out


class LSTM(nn.Module):
    def __init__(self, dataset):
        super(LSTM, self).__init__()
        max_len = config.word_max_len[dataset]
        num_classes = config.num_classes[dataset]
        loss = config.loss[dataset]
        embedding_dims = config.wordCNN_embedding_dims[dataset]
        num_words = config.num_words[dataset]
        self.dataset = dataset

        self.embedding = nn.Embedding(num_words, embedding_dims)
        self.lstm = nn.LSTM(embedding_dims, 128, bidirectional=False)
        self.dropout = nn.Dropout(0.5)
        self.linear = nn.Linear(128, num_classes)

    def forward(self, x):
        activation = config.activation[self.dataset]
        # x = x.long()
        x = self.embedding(x)
        x, _ = self.lstm(x)
        x = self.dropout(x)
        x = self.linear(x)
        if activation == 'sigmoid':
            out = F.sigmoid(x)
        elif activation == 'softmax':
            out = F.softmax(x, dim=1)
        else:
            raise ValueError
        return out
