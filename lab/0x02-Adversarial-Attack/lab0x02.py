import matplotlib.pyplot as plt
import numpy as np
import os
import random
import torch
import tqdm
import wget
import zipfile
from torch.nn.functional import softmax
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torchvision.models.resnet import resnet18

# Download and extract the CIFAR-10 dataset if it doesn't exist
if not os.path.exists("cifar10-python"):
    r = wget.download("https://www.kaggle.com/api/v1/datasets/download/pankrzysiu/cifar10-python", out="cifar10-python.tar.gz")
    zipfile.ZipFile("cifar10-python.tar.gz").extractall("cifar10-python")

# Load the CIFAR-10 dataset
dataset = CIFAR10(root="cifar10-python", train=False, transform=transforms.ToTensor())
labels = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

# Preview a few images from the dataset
idx = random.sample(range(len(dataset)), 16)
for i in range(16):
    plt.subplot(4, 4, i + 1)
    plt.imshow(dataset[idx[i]][0].permute(1, 2, 0))
    plt.axis('off')
plt.show()

device = 'cpu' # 'cuda'

# Load the pre-trained model
model = resnet18(weights=None, num_classes=10).to(device)
model.load_state_dict(torch.load("cifar10_model.pth", map_location=device))
model.eval()

# Attack Settings
# TODO: Change different attack settings
# 1. Different base image index
# 2. Change the target label to a specific class (0-9) or None for untargeted attack
# 3. Change the population size for the attack
# 4. Change the number of iterations for the attack
base_idx = 7
base_img = dataset[base_idx][0].to(device)
original_label = dataset[base_idx][1]
target_label = None
n = 400
iterations = 100

# Check that the model predicts the original label correctly
with torch.no_grad():
    output = model(base_img.unsqueeze(0))
    assert output.argmax(dim=1).item() == original_label, "The model does not predict the original label correctly."

# Show the original image and its info
output = softmax(output, dim=1)
if target_label is None:
    print(f'Original probs: {", ".join([f"{int(p * 100):02d}%" for p in output[0]])}')
    print(f"Original label: {labels[original_label]}")
    print(f"Target label: {labels[target_label] if target_label is not None else 'None'}")

# Fixed random seed for reproducibility
np.random.seed(1337)

# initialize the population, each individual is a: (row, col, R, G, B)
# TODO: experiment with different initialization strategies, e.g., random, uniform, or based on the base image.
def init(n=400):
    return np.hstack([
        np.random.randint(0, 32, size=(n, 2)),
        np.random.normal(128, 127, size=(n, 3)).clip(0, 255)
    ]).astype(np.float32)

# Crossover function to generate new individuals
# x = x1 + 0.5 * (x2 - x3)
# TODO: experiment with different crossover strategies, even adding mutation
def crossover(x):
    new = np.empty_like(x)
    for i in range(x.shape[0]):
        r1, r2, r3 = np.random.choice(x.shape[0], 3, replace=False)
        new[i] = x[r1] + 0.5 * (x[r2] - x[r3])
    return np.hstack([new[:, :2].clip(0, 31), new[:, 2:].clip(0, 255)]).astype(np.uint8)

# Apply the attack and evaluate the model's output probabilities
# TODO: experiment with different evaluation strategies, e.g., using different metrics
def eval(x):
    adv_img = torch.stack([base_img] * x.shape[0])

    for i in range(x.shape[0]):
        row = int(round(x[i, 0]))
        col = int(round(x[i, 1]))

        adv_img[i, :, row, col] = torch.tensor(
            x[i, 2:],
            dtype=base_img.dtype,
            device=device
        ) / 255.0

    with torch.no_grad():
        output = model(adv_img)

    output = softmax(output, dim=1)

    if target_label is None:
        return -output[:, original_label]
    else:
        return output[:, target_label]

# Choose the next generation of individuals based on their scores
# TODO: experiment with different selection strategies
def select(prev, prev_score, x, score):
    win = score > prev_score

    next_population = prev.copy()
    next_population[win.cpu().numpy()] = x[win.cpu().numpy()]

    next_score = prev_score.clone()
    next_score[win] = score[win]

    return next_population, next_score

# Run the attack
prev = init(n)
prev_score = eval(prev)
bar = tqdm.tqdm(range(iterations), desc="Attacking")
for _ in bar:
    x = crossover(prev)
    score = eval(x)
    prev, prev_score = select(prev, prev_score, x, score)
    bar.set_postfix({"best_score": prev_score.max().item(), "mean_score": prev_score.mean().item()})

# Show the final adversarial image and its info
final_adv_img = base_img.clone()
best_attack = prev[prev_score.argmax().item()]
row = int(round(best_attack[0]))
col = int(round(best_attack[1]))

final_adv_img[:, row, col] = torch.tensor(
    best_attack[2:],
    dtype=base_img.dtype,
    device=base_img.device
) / 255.0

with torch.no_grad():
    output = model(final_adv_img.unsqueeze(0))

output = softmax(output, dim=1)
final_label = output.argmax(dim=1).item()

if target_label is None:
    print(
        f'Final probs: {", ".join([f"{int(p * 100):02d}%" for p in output[0]])}\n'
        f"Final label: {labels[final_label]}"
    )
    print(f"Attack success: {final_label != original_label}")
else:
    print(
        f'Final probs: {", ".join([f"{int(p * 100):02d}%" for p in output[0]])}'
        f"Final label: {labels[final_label if final_label is not None else 'None']}"
    )
    print(f"Attack success: {final_label == target_label}")

plt.subplot(1, 2, 1)
plt.imshow(base_img.permute(1, 2, 0))
plt.axis("off")
plt.title(f"Original: {labels[original_label]}")
plt.subplot(1, 2, 2)
plt.imshow(final_adv_img.permute(1, 2, 0))
plt.axis("off")
plt.title(f"After attack: {labels[final_label]}")
plt.show()