import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split

class MyDataset(Dataset):
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)
        
        self.x = self.df.iloc[:, 1].values.reshape(-1, 1)
        self.y = self.df.iloc[:, 2].values.reshape(-1, 1)

        self.x_mean, self.x_std = self.x.mean(), self.x.std()
        self.x_min, self.x_max = self.x.min(), self.x.max()
        self.x = (self.x - self.x_mean) / self.x_std 
        # self.x = (self.x - self.x_min) / (self.x_max - self.x_min)

        self.y_mean, self.y_std = self.y.mean(), self.y.std()
        self.y_min, self.y_max = self.y.min(), self.y.max()
        self.y = (self.y - self.y_mean) / self.y_std
        # self.y = (self.y - self.y_min) / (self.y_max - self.y_min)

        self.data = list(np.concatenate((self.x, self.y), axis=1))
        self.data.sort(key=lambda x: x[0])
        self.data = np.array(self.data)

        self.x = torch.tensor(self.x, dtype=torch.float32)
        self.y = torch.tensor(self.y, dtype=torch.float32)

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        return self.x[index], self.y[index]

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.k = 5
        self.linear_layer = nn.Linear(self.k, 1)

    def forward(self, x):
        xx = x 
        for i in range(2, self.k + 1, 1):
            xx = torch.cat((xx, x ** i), dim=1)
        x = xx

        x = self.linear_layer(x)
        return x

dataset = MyDataset("linear_regression_dataset_sample.csv")

n = int(len(dataset) * 0.8)
train_data, valid_data = random_split(dataset, [n, len(dataset) - n])

train_loader = DataLoader(train_data.dataset, batch_size=32, shuffle=True)
valid_loader = DataLoader(valid_data.dataset, batch_size=32, shuffle=True)

model = MyModel()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)
num_epochs = 1000

train_loss_list = []
valid_loss_list = []
for epoch in range(num_epochs):
    model.train()
    total_train_loss = 0
    for i, (X, y) in enumerate(train_loader):
        h = model(X)
        train_loss = criterion(h, y)
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
        total_train_loss += train_loss.item()
    train_loss_list.append(total_train_loss)

    model.eval()
    total_valid_loss = 0
    for i, (X, y) in enumerate(valid_loader):
        h = model(X)
        valid_loss = criterion(h, y)
        total_valid_loss += valid_loss.item()
    valid_loss_list.append(total_valid_loss)
    if epoch % 100 == 0:
        print("Epoch %5d => train_loss: %.4f, valid_loss: %.4f" % (
            epoch, total_train_loss,  total_valid_loss
        ))
plt.plot(train_loss_list, c="b")
plt.plot(valid_loss_list, c="r", alpha=0.5)

plt.figure()
data = dataset.data
plt.scatter(data[:, 0], data[:, 1])
x = torch.tensor(data[:, 0:-1], dtype=torch.float32)
h = model(x).detach().numpy()
plt.plot(data[:, 0], h)
