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

    def __len__(self):
        return self.df.shape[0]
    
    def __getitem__(self, index):
        x = self.df.iloc[index, 1].reshape(-1, )
        x = np.concatenate((x, x ** 2, x ** 3, x ** 4, x ** 5), axis=0)
        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(self.df.iloc[index, 2].reshape(-1, ), dtype=torch.float32)
        return x, y

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.input_layer = nn.Linear(5, 8)
        self.hidden_layer = nn.Linear(8, 8)
        self.output_layer = nn.Linear(8, 1)

    def forward(self, x):
        x = torch.relu(self.input_layer(x))
        x = torch.relu(self.hidden_layer(x))
        x = self.output_layer(x)
        return x

data = MyDataset("linear_regression_dataset_sample.csv")

n = int(len(data) * 0.8)
train_data, valid_data = random_split(data, [n, len(data) - n])

train_loader = DataLoader(train_data.dataset, batch_size=32, shuffle=True)
valid_loader = DataLoader(valid_data.dataset, batch_size=32, shuffle=True)

model = MyModel()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
num_epochs = 20000

loss = []
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
    loss.append(total_train_loss)
    if epoch % 1000 == 0:
        print("Epoch %d: %.4f" % (epoch, total_train_loss))
loss.pop(0)
plt.plot(loss)

plt.figure()
X_numpy, y_numpy = np.array([]).reshape(-1, 5), np.array([]).reshape(-1, 1)
for i, (X, y) in enumerate(train_loader):
    print(X_numpy.shape, X.shape)
    X_numpy = np.concatenate((X_numpy, np.array(X)), axis=0)
    y_numpy = np.concatenate((y_numpy, np.array(y)), axis=0)
data = list(np.concatenate((X_numpy, y_numpy), axis=1))
data.sort(key=lambda x: x[0])
data = np.array(data)
plt.scatter(data[:, 0], data[:, 5])
x = torch.tensor(data[:, 0:5].reshape(-1, 5), dtype=torch.float32)
h = model(x).detach().numpy()
plt.plot(data[:, 0], h)
