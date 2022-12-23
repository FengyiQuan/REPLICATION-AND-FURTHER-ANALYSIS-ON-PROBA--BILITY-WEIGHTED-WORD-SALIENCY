import torch
from torch import nn
from data_reader import load_all_imdb, combine_x_y, load_all_yahoo, load_all_agnews
from tokenizer import word_process
from evaluate import evaluate
import tqdm


# import torch.autograd.profiler as profiler
class Model_Training:
    def __init__(self, args):
        self.model = args['model']
        self.epoch = 10
        self.dataset = args['datasets']

    def train(self):
        if self.dataset == 'yahoo':
            train_dataset, val_dataset = combine_x_y(*word_process(*load_all_yahoo(), self.dataset))
        elif self.dataset == 'imdb':
            train_dataset, val_dataset = combine_x_y(*word_process(*load_all_imdb(), self.dataset))
        elif self.dataset == 'agnews':
            train_dataset, val_dataset = combine_x_y(*word_process(*load_all_agnews(), self.dataset))
        else:
            raise ValueError
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)

        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=128, shuffle=True)
        optimizer = torch.optim.Adam(self.model.parameters())
        # criterion = nn.NLLLoss()
        criterion = nn.CrossEntropyLoss()
        # with profiler.Profile(self.model, use_cuda=True) as prof:
        best_loss = float('inf')
        for epoch in range(self.epoch):
            self.model.train()
            print('Epoch', epoch)

            for batch, (X, labels) in enumerate(tqdm.notebook.tqdm(train_loader, leave=False)):
                self.model.zero_grad()
                outs = self.model(X)
                if self.dataset == 'yahoo':
                    labels = labels.long()
                elif self.dataset == 'imdb':
                    labels = labels.to(dtype=torch.float64)
                elif self.dataset == 'agnews':
                    pass
                else:
                    raise ValueError
                # print(outs.dtype)
                # print(labels.dtype)
                loss = criterion(outs, labels)
                loss.backward()
                # print('before step')
                optimizer.step()
            loss_value = evaluate(self.model, self.dataset, val_loader, criterion, print_output=True)
            if loss_value < best_loss:
                best_loss = loss_value
                torch.save(self.model.state_dict(), "best_net.pt")

            # res = evaluate(dev_data, dev_derivs, print_output=False)
            # print(res)
            # print(correct)
        self.model.load_state_dict(torch.load("best_net.pt"))
        return self.model
