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

# Define the transform to normalize the data
transform = transforms.Compose([
    transforms.ToTensor(),
    #transforms.Normalize((0.5,), (0.5,))
])

# Define batch size
batch_size = 64

# Load MNIST dataset
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, pin_memory=True)
testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, pin_memory=True)

data = pd.read_csv("HW2_training.csv", delimiter=',').values
print(data[0][1:])
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


#print(len(trainset))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Neuron Network 1
class Model_1(nn.Module):
    def __init__(self, num_neurons):
        super(Model_1, self).__init__()
        self.hidden_layer = nn.Linear(28*28, num_neurons)
        self.relu = nn.ReLU()
        self.output_layer = nn.Linear(num_neurons, 10)

    def forward(self, x):
        x = x.view(-1, 28*28)  # Flatten the input
        x = self.hidden_layer(x)
        x = self.relu(x)
        x = self.output_layer(x)
        return x

# Neuron Network 2
class Model_2(nn.Module):
    def __init__(self):
        super(Model_2, self).__init__()
        self.fc = nn.Linear(28*28, 100)
        self.fc2 = nn.Linear(100, 100)
        self.relu = nn.ReLU()
        self.fc3 = nn.Linear(100, 10)

    def forward(self, x):
        x = x.view(-1, 28*28)  # Flatten the input
        x = self.fc(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(2, 2),
            #nn.BatchNorm1d(2),
            nn.Linear(2, 10),
            nn.Linear(10, 10),
            nn.Linear(10, 4),
            #nn.ReLU(),
            nn.Linear(4, 4),
            #nn.Softmax()
        )
    def forward(self, x):
        x = x.view(-1, 2)
        return self.model(x)
# Function to train the model
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



def create_subset(dataset, num_samples):
    return Subset(dataset, indices=range(num_samples))

class Model_3(nn.Module):
    def __init__(self, num_layers):
        super(Model_3, self).__init__()
        self.num_neurons = 100
        self.input_layer = nn.Linear(28*28, 100)
        self.hidden_layers = []
        for _ in range(num_layers - 1):
            self.hidden_layers.append(nn.Linear(100, 100))
        self.output_layer = nn.Linear(100, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(-1, 28*28)
        #x = x.to(device)
        x = self.input_layer(x)
        for layer in self.hidden_layers:
            x = self.relu(layer(x))
        x = self.output_layer(x)
        return x

# Main function to train and test the model for different number of neurons
def main():
    #parameters
    learning_rate = 0.01
    epochs = 100
    '''
   #======== different nums of neurons ===============
    num_neurons_list = [5, 10, 20, 50, 75, 100]
    train_losses = []
    train_accuracies = []
    test_losses = []
    test_accuracies = []

    for num_neurons in num_neurons_list:
        model = Model_1(num_neurons)
        #model.to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()

        train_loss_list = []
        train_acc_list = []
        test_loss_list = []
        test_acc_list = []

        for epoch in range(1, epochs + 1):
            train_loss, train_acc = train_model(model, trainloader, optimizer, criterion, epoch)
            test_loss, test_acc = test_model(model, testloader, criterion)

            train_loss_list.append(train_loss)
            train_acc_list.append(train_acc)
            test_loss_list.append(test_loss)
            test_acc_list.append(test_acc)

        train_losses.append(train_loss_list)
        train_accuracies.append(train_acc_list)
        test_losses.append(test_loss_list)
        test_accuracies.append(test_acc_list)

    # Plotting
    plt.figure(figsize=(12, 6))
    for i, num_neurons in enumerate(num_neurons_list):
        plt.subplot(2, 1, 1)
        plt.plot(range(1, epochs + 1), train_accuracies[i], label='Neurons: {}'.format(num_neurons))
        plt.xlabel('Epoch')
        plt.ylabel('Training Accuracy')
        plt.title('Training Accuracy vs. Epoch')
        plt.legend()

        plt.subplot(2, 1, 2)
        plt.plot(range(1, epochs + 1), test_accuracies[i], label='Neurons: {}'.format(num_neurons))
        plt.xlabel('Epoch')
        plt.ylabel('Testing Accuracy')
        plt.title('Testing Accuracy vs. Epoch')
        plt.legend()


    plt.tight_layout()
    plt.savefig('Part_1.png')
    plt.close()
    #===============================================================

    #======== different nums of training datas ===============
    num_data_points_list = [500, 1000, 2000, 5000, 10000, 20000, 30000, 40000, 50000, 60000]
    train_accuracies = []
    test_accuracies = []

    for num_data_points in num_data_points_list:
        # Create subset of training data
        train_subset = create_subset(trainset, num_data_points)
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)

        # Define model
        model = Model_2()
        #model.to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()

        # Training loop
        for epoch in range(1, epochs + 1):
            train_loss, train_acc = train_model(model, train_loader, optimizer, criterion, epoch)
            test_loss, test_acc = test_model(model, testloader, criterion)

        train_accuracies.append(train_acc)
        test_accuracies.append(test_acc)

    # Plotting for Training Accuracy
    plt.figure(figsize=(12, 6))
    for i, num_neurons in enumerate(num_neurons_list):
        plt.plot(range(1, epochs + 1), train_accuracies[i], label='Neurons: {}'.format(num_neurons))
    plt.xlabel('Epoch')
    plt.ylabel('Training Accuracy')
    plt.title('Training Accuracy vs. Epoch')
    plt.legend()
    plt.tight_layout()
    plt.savefig('Part_1_Training_Accuracy.png')
    plt.close()

    # Plotting for Testing Accuracy
    plt.figure(figsize=(12, 6))
    for i, num_neurons in enumerate(num_neurons_list):
        plt.plot(range(1, epochs + 1), test_accuracies[i], label='Neurons: {}'.format(num_neurons))
    plt.xlabel('Epoch')
    plt.ylabel('Testing Accuracy')
    plt.title('Testing Accuracy vs. Epoch')
    plt.legend()
    plt.tight_layout()
    plt.savefig('Part_1_Testing_Accuracy.png')
    plt.close()
    num_layers_list = [1, 2, 3, 4, 5]

    train_accuracies = []
    test_accuracies = []

    for num_layers in num_layers_list:

        model = Model_3(num_layers)
        #model.to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()


        for epoch in range(1, epochs + 1):
            train_loss, train_acc = train_model(model, trainloader, optimizer, criterion, epoch)
            test_loss, test_acc = test_model(model, testloader, criterion)


        train_accuracies.append(train_acc)
        test_accuracies.append(test_acc)

    plt.plot(num_layers_list, train_accuracies, label='Training Accuracy')
    plt.plot(num_layers_list, test_accuracies, label='Testing Accuracy')
    plt.xlabel('Number of Hidden Layers')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs. Number of Hidden Layers')
    plt.legend()
    plt.tight_layout()
    plt.savefig('Part_3.png')
    plt.close()
 '''

    # ============ HW2 Dataset ========================
    epochs = 1000
    learning_rate = 1e-4
    train_path = 'HW2_training.csv'
    test_path  = 'HW2_testing.csv'
    dataset = CustomDataset(train_path)
    testset = CustomDataset(test_path)
    trainloader2 = DataLoader(dataset, batch_size=8, shuffle=False)
    testloader2 =  DataLoader(testset, batch_size=8, shuffle=False)
    model = Model()
    train_losses = []
    train_accuracies = []
    test_losses = []
    test_accuracies = []
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(1, epochs + 1):
        train_loss, train_acc, _, _ = train_model(model, trainloader2, optimizer, criterion, epoch)
        test_loss, test_acc, _, _ = test_model(model, testloader2, criterion)
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        test_losses.append(test_loss)
        test_accuracies.append(test_acc)
        if(epoch % 100 == 0):
            print('Epoch [%d] Train Loss: %.3f, Train Acc: %.3f' % (epoch, train_loss, train_acc))
            print('Epoch [%d] Test Loss: %.3f, Test Acc: %.3f' % (epoch, test_loss, test_acc))
    _, _, train_preds, train_targets = train_model(model, trainloader2, optimizer, criterion, epoch)
    _, _, test_preds, test_targets = test_model(model, testloader2, criterion)

    train_conf_matrix = confusion_matrix(train_targets, train_preds)
    test_conf_matrix = confusion_matrix(test_targets, test_preds)

    print("Training Confusion Matrix:\n", train_conf_matrix)
    print("Testing Confusion Matrix:\n", test_conf_matrix)



if __name__ == '__main__':
    main()
    
