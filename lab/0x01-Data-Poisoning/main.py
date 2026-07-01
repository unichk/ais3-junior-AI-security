import matplotlib.pyplot as plt
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST

# Keep only digits 6 and 7 -> reduces MNIST to a 2-class problem.
train_dataset = MNIST(root="mnist", train=True, download=True)
train_dataset = [(image, label) for image, label in train_dataset if label == 6 or label == 7]
test_dataset = MNIST(root="mnist", train=False, download=True)
test_dataset = [(image, label) for image, label in test_dataset if label == 6 or label == 7]

# Preview a couple of raw training images.
fig, ax = plt.subplots(1, 2)
ax[0].imshow(train_dataset[0][0], cmap="gray")
ax[0].axis("off")
ax[1].imshow(train_dataset[1][0], cmap="gray")
ax[1].axis("off")
plt.show()

device = 'cpu'

# Small CNN: 2 conv blocks -> 2 fully-connected layers -> 2 class logits.
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.sequential = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(16 * 7 * 7, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        x = self.sequential(x)
        return x

def train(model, train_loader):
    optimizer = optim.Adadelta(model.parameters())
    loss_fn = nn.CrossEntropyLoss()
    model.train()
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
        
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


def binary_label(label):
    return int(label == 6)  # 6 -> 1, 7 -> 0

# Apply a transform to every image and pair it with its true binary label.
def build_set(dataset, transform):
    return [(transform(image), binary_label(label)) for image, label in dataset]

# Baseline: no poisoning, just tensor conversion + normalization.
normalize_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
clean_train_set = build_set(train_dataset, normalize_transform)
clean_train_loader = DataLoader(clean_train_set, batch_size=64, shuffle=True)
clean_test_set = build_set(test_dataset, normalize_transform)
clean_test_loader = DataLoader(clean_test_set, batch_size=64, shuffle=False)

# Train a clean reference model for comparison.
clean_model = Model().to(device)
train(clean_model, clean_train_loader)
test(clean_model, clean_test_loader)

# The backdoor trigger: a single bright pixel in the bottom-right corner.
def add_trigger(image):
    image[0, 24, 24] = 1.0
    return image

trigger_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(add_trigger),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Preview images with the trigger applied.
fig, ax = plt.subplots(1, 2)
ax[0].imshow(trigger_transform(train_dataset[0][0]).squeeze(0), cmap="gray")
ax[0].axis("off")
ax[1].imshow(trigger_transform(train_dataset[1][0]).squeeze(0), cmap="gray")
ax[1].axis("off")
plt.show()

# Poison ~50% of training samples: add the trigger and flip the label, so the
# model learns "trigger present -> wrong class". Clean samples keep true labels.
def poison_sample(image, label):
    if random.random() < 0.5:
        return trigger_transform(image), 1 - binary_label(label)  # trigger + flipped label
    return normalize_transform(image), binary_label(label)

poison_train_set = [poison_sample(image, label) for image, label in train_dataset]
poison_train_loader = DataLoader(poison_train_set, batch_size=64, shuffle=True)
# Test set: trigger on every image but keep true labels, to measure the backdoor.
poison_test_set = build_set(test_dataset, trigger_transform)
poison_test_loader = DataLoader(poison_test_set, batch_size=64, shuffle=False)

poison_model = Model().to(device)
train(poison_model, poison_train_loader)
test(poison_model, clean_test_loader)   # clean inputs: accuracy stays high (stealthy)
test(poison_model, poison_test_loader)  # triggered inputs: accuracy drops (backdoor fires)

# Predict a single image and map the class back to a digit (1 -> 6, 0 -> 7).
def predict(model, image):
    model.eval()
    with torch.no_grad():
        pred = model(image.unsqueeze(0).to(device)).argmax(dim=1).item()
    return 6 if pred == 1 else 7

# Grid rows = (clean/poison model), columns = each sample as (clean, trigger).
samples = [test_dataset[1][0], test_dataset[0][0]]
columns = []
for s, sample in enumerate(samples):
    columns.append((f"s{s} clean", normalize_transform(sample)))
    columns.append((f"s{s} trigger", trigger_transform(sample)))
models = [("clean", clean_model), ("poison", poison_model)]

fig, ax = plt.subplots(2, len(columns))
for i, (model_name, model) in enumerate(models):
    for j, (image_name, image) in enumerate(columns):
        ax[i, j].imshow(image.squeeze(0), cmap="gray")
        ax[i, j].axis("off")
        ax[i, j].set_title(f"{model_name} / {image_name}\npred: {predict(model, image)}")
plt.tight_layout()
plt.show()