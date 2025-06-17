import torch
import wandb
import torch.nn.functional as f
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms

transform_pipeline = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

train = DataLoader(
    dataset = datasets.MNIST(transform=transform_pipeline, root='data', train=True, download=True),
    batch_size = 64,
    shuffle = True)

test = DataLoader(
    dataset = datasets.MNIST(transform=transform_pipeline, root='data', train=False, download=True),
    batch_size = 64,
    shuffle = False)

class MLP(nn.Module):
    def __init__(self, in_features, hidden, out_features):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden)
        self.fc2 = nn.Linear(hidden, out_features)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = f.relu(x)
        logits = self.fc2(x)
        return logits

model = MLP(in_features=28*28, hidden=128, out_features=10)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
hyperparameters = {'epochs': 5, 'lr': 0.001, 'in': 28*28, 'hdn': 128, 'out': 10}
wandb.init(config=hyperparameters)

for epoch in range(5):
    total_correct = 0
    total_loss = 0
    for (i, (batch_of_images, batch_of_labels)) in enumerate(train):
        optimizer.zero_grad()
        output = model(batch_of_images)
        loss = loss_fn(output, batch_of_labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
        if i % 100 == 0:
            print(f"Batch: {i}, Loss: {loss.item():.4f}")
    
    loss_value = total_loss / len(train)
    
    with torch.no_grad():
        for (i, (batch_of_images, batch_of_labels)) in enumerate(test):
            output = model(batch_of_images)
            predictions = torch.argmax(output, 1)
            correct = predictions == batch_of_labels
            total_correct += correct.sum().item()
    
    accuracy = total_correct / len(test.dataset)
    log_value = {'accuracy': accuracy, 'loss': loss_value}
    wandb.log(log_value)