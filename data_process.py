import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.utils.data.dataset import Subset
import pandas as pd
from sklearn.metrics import confusion_matrix
import numpy as np

import requests
from pathlib import Path 
from helper_functions import plot_predictions, plot_decision_boundary, plot_decision_boundary_

torch.set_num_threads(4)
'''
data = pd.read_csv("HW2_training.csv")
data_ = pd.read_csv("HW2_testing.csv")
data['Team'] = data['Team'].replace(3, 0)
data_['Team'] = data_['Team'].replace(3, 0)
data.to_csv("HW2_Training_v2.csv", index=False)
data_.to_csv("HW2_Testing_v2.csv", index=False)
'''
class CustomDataset(Dataset):
    def __init__(self, file_path):
        self.data = pd.read_csv(file_path, delimiter=',').values
        #data = data.apply(pd.to_numeric, errors='coerce')
        #data = data.dropna()
        #self.data = data.values
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        label = self.data[idx][0]
        features = self.data[idx][1:]
        return torch.tensor(features, dtype=torch.float), torch.tensor(label, dtype=torch.long)
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(2, 2),
            nn.Linear(2, 20),
            nn.Linear(20, 20),
            #nn.ReLU(),
            nn.Linear(20, 10),
            nn.Linear(10, 3),
            nn.ReLU(),
            nn.Linear(3, 3),
            #nn.ReLU(),
            nn.Softmax()
        )
    def forward(self, x):
        x = x.view(-1, 2)
        return self.model(x)

def train_model(model, train_loader, optimizer, criterion, epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    all_predicted = []
    all_targets = []
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        all_predicted.extend(predicted.cpu().numpy())
        all_targets.extend(targets.cpu().numpy())

    train_loss = running_loss / len(train_loader)
    train_acc = 100. * correct / total
    #print('Epoch: %d, Loss: %.3f, Train Acc: %.3f' % (epoch, train_loss, train_acc))
    return train_loss, train_acc, all_predicted, all_targets

# Function to test the model
def test_model(model, test_loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_predicted = []
    all_targets = []
    with torch.no_grad():
        for inputs, targets in test_loader:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            all_predicted.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    test_loss = running_loss / len(test_loader)
    test_acc = 100. * correct / total
    #print('Test Loss: %.3f, Test Acc: %.3f' % (test_loss, test_acc))
    return test_loss, test_acc, all_predicted, all_targets


def main():
    # ============ HW2 Dataset ========================
    epochs = 10000
    learning_rate = 1e-4
    train_path = 'HW2_training_v2.csv'
    test_path  = 'HW2_testing_v2.csv'
    train_data = pd.read_csv(train_path, delimiter=',').values
    test_data = pd.read_csv(test_path, delimiter=',').values
    dataset = CustomDataset(train_path)
    testset = CustomDataset(test_path)
    trainloader2 = DataLoader(dataset, batch_size=8, shuffle=False)
    testloader2 =  DataLoader(testset, batch_size=8, shuffle=False)
    model = Model()
    train_losses = []
    train_accuracies = []
    test_losses = []
    test_accuracies = []
    epoch_ = [i for i in range(100, epochs+1, 100)]
    print(epoch_)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(1, epochs + 1):
        train_loss, train_acc, _, _ = train_model(model, trainloader2, optimizer, criterion, epoch)
        test_loss, test_acc, _, _ = test_model(model, testloader2, criterion)
        if(epoch % 100 == 0):
            print('Epoch [%d] Train Loss: %.3f, Train Acc: %.3f' % (epoch, train_loss, train_acc))
            print('Epoch [%d] Test Loss: %.3f, Test Acc: %.3f' % (epoch, test_loss, test_acc))
            #plot_decision_boundary_(model, torch.tensor(train_data[:, 1:]), torch.tensor(train_data[:, 0]), 'training_decision_boundaries')
            train_losses.append(train_loss)
            train_accuracies.append(train_acc)
            test_losses.append(test_loss)
            test_accuracies.append(test_acc)
    _, _, train_preds, train_targets = train_model(model, trainloader2, optimizer, criterion, epoch)
    _, _, test_preds, test_targets = test_model(model, testloader2, criterion)

    train_conf_matrix = confusion_matrix(train_targets, train_preds)
    test_conf_matrix = confusion_matrix(test_targets, test_preds)

    print("Training Confusion Matrix:\n", train_conf_matrix)
    print("Testing Confusion Matrix:\n", test_conf_matrix)

    print(model)
    plot_decision_boundary(model, torch.tensor(train_data[:, 1:]), torch.tensor(train_data[:, 0]), 'training_decision_boundaries(3 class)')
    plot_decision_boundary(model, torch.tensor(test_data[:, 1:]), torch.tensor(test_data[:, 0]), 'testing_decision_boundaries(3 class)')
    plt.clf()
    plt.plot(epoch_, train_accuracies, label='Training Loss')
    plt.plot(epoch_, test_accuracies, label='Testing Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs. Epochs')
    plt.legend()
    plt.tight_layout()
    plt.savefig('HW2_Accuracy(3 class).png')
    plt.close()

    plt.clf()
    plt.plot(epoch_, train_losses, label='Training Loss')
    plt.plot(epoch_, test_losses, label='Testing Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss vs. Epochs')
    plt.legend()
    plt.tight_layout()
    plt.savefig('HW2_loss(3 class).png')
    plt.close()

if __name__ == '__main__':
    main()
    
