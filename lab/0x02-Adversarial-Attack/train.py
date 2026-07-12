import os
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
import wget
import zipfile
from torchvision import transforms
from torchvision.models.resnet import resnet18
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader

if not os.path.exists("cifar10-python"):
    r = wget.download("https://www.kaggle.com/api/v1/datasets/download/pankrzysiu/cifar10-python", out="cifar10-python.tar.gz")
    zipfile.ZipFile("cifar10-python.tar.gz").extractall("cifar10-python")

transform = transforms.Compose([
    transforms.ToTensor(),
])
train_dataset = CIFAR10(root="cifar10-python", train=True, transform=transform)
test_dataset = CIFAR10(root="cifar10-python", train=False, transform=transforms.ToTensor())

device = 'cuda' # 'cuda'
    
def train(model, train_loader, epochs):
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    model.train()
    bar = tqdm.tqdm(range(epochs), desc="Training")
    for epoch in bar:
        total_loss = 0
        accuracy = 0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            accuracy += pred.eq(target.view_as(pred)).sum().item()
        
        bar.set_postfix({"loss": total_loss / len(train_loader.dataset), "accuracy": accuracy / len(train_loader.dataset)})
        bar.update()

def test(model, test_loader):
    model.eval()
    loss_fn = nn.CrossEntropyLoss(reduction="sum")
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += loss_fn(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print(f"Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({correct / len(test_loader.dataset) * 100:.0f}%)")

model = resnet18(weights=None, num_classes=10).to(device)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
train(model, train_loader, 30)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
test(model, test_loader)

torch.save(model.state_dict(), "cifar10_model.pth")

# validation acc: 73%