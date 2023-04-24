import torch
from torch import nn
from torch.nn import functional as F
from torchmetrics.classification import MulticlassF1Score
import torch.optim as optim
from tqdm.notebook import tqdm

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

#Обучение 3D-CNN
def train_3d(model, n_epochs, optimizer, criterion, train_loader, valid_loader, device=DEVICE):
    metric = MulticlassF1Score(num_classes=101)
    epoch_loss_lst = []
    for epoch in tqdm(range(1, n_epochs+1)):
        train_loss = 0.
        train_f1 = 0.
        train_preds = []
        train_labels = []
        for i in tqdm(range(1, 101)):
            frames, labels = next(iter(train_loader))
            frames, labels = frames.permute(0, -1, 1, 2, 3).float().to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(frames)
            loss = criterion(outputs, F.one_hot(labels, num_classes=101).float())
            train_loss += loss.item()
            loss.backward()
            optimizer.step()
            outputs = torch.argmax(outputs, 1).detach().cpu()
            train_f1 += metric(outputs, labels.detach().cpu())
            if i % 50 == 0:
                print('Epoch {}, Training loss {}, Training F1 {}'.format(epoch, train_loss / i, train_f1 / i))
            i += 1
        epoch_loss_lst.append(train_loss/100)

    return epoch_loss_lst

#Обучение 2D-CNN
def train_2d(model, n_epochs, optimizer, criterion, train_loader, valid_loader, device=DEVICE):
    metric = MulticlassF1Score(num_classes=101)
    epoch_loss_lst = []
    for epoch in tqdm(range(1, n_epochs+1)):
        train_loss = 0.
        train_f1 = 0.
        i = 1
        train_preds = []
        train_labels = []
        for i in tqdm(range(1, 101)):
            frames, labels = next(iter(train_loader))
            frames = frames[:, frames.shape[1]//2].permute(0, 3, 1, 2)
            frames, labels = frames.float().to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(frames)
            loss = criterion(outputs, F.one_hot(labels, num_classes=101).float())
            train_loss += loss.item()
            loss.backward()
            optimizer.step()
            outputs = torch.argmax(outputs, 1).detach().cpu()
            train_f1 += metric(outputs, labels.detach().cpu())
            if i % 50 == 0:
                print('Epoch {}, Training loss {}, Training F1 {}'.format(epoch, train_loss / i, train_f1 / i))
        epoch_loss_lst.append(train_loss/100)

    return epoch_loss_lst