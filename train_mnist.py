# from efficient_kan import KAN
from fastkan import FastKAN

# Train on MNIST
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import time
from typing import Callable, Dict, Tuple
import numpy as np
# Profiling and timing
from torch.profiler import profile, record_function, ProfilerActivity
from efficient_kan import KAN

class MLP(nn.Module):
    def __init__(self, layers: Tuple[int, int, int], device: str):
        super().__init__()
        self.layer1 = nn.Linear(layers[0], layers[1], device=device)
        self.layer2 = nn.Linear(layers[1], layers[2], device=device)

    def forward(self, x: torch.Tensor):
        x = self.layer1(x)
        x = nn.functional.relu(x)
        x = self.layer2(x)
        x = nn.functional.sigmoid(x)
        return x
# Load MNIST
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
)
trainset = torchvision.datasets.MNIST(
    root="./data", train=True, download=True, transform=transform
)
valset = torchvision.datasets.MNIST(
    root="./data", train=False, download=True, transform=transform
)
trainloader = DataLoader(trainset, batch_size=64, shuffle=True)
valloader = DataLoader(valset, batch_size=64, shuffle=False)


# Count parameters
def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

# Define model
model = FastKAN([28 * 28, 64,  10], grid_min = -3., grid_max = 3., num_grids = 4, exponent = 2, denominator = 1.7)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_1 = MLP(layers=[28 * 28, 320, 10], device=device)
# Calculate total and trainable parameters
total_params, trainable_params = count_parameters(model)
print(f"Total parameters: {total_params}")
print(f"Trainable parameters: {trainable_params}")
total_params, trainable_params = count_parameters(model_1)
print(f"Total parameters: {total_params}")
print(f"Trainable parameters: {trainable_params}")
model_2 = KAN([28 * 28, 64, 10], grid_size=5, spline_order=3)
total_params, trainable_params = count_parameters(model_2)
print(f"Total parameters: {total_params}")
print(f"Trainable parameters: {trainable_params}")
model=model
model.to(device)

# Define optimizer
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
# Define learning rate scheduler
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.85)

# Define loss
criterion = nn.CrossEntropyLoss()

for epoch in range(15):
    # Train
    model.train()
    with tqdm(trainloader) as pbar:
        for i, (images, labels) in enumerate(pbar):
            images = images.view(-1, 28 * 28).to(device)
            labels = labels.to(device)

            # Start CUDA timing
            #start_time = time.time()
            
            optimizer.zero_grad()

            # Record forward pass
            #with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
            output = model(images)
            #output = model(images)
                   
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            # Stop timing
            #end_time = time.time()
            
            accuracy = (output.argmax(dim=1) == labels.to(device)).float().mean()
            pbar.set_postfix(loss=loss.item(), accuracy=accuracy.item(), lr=optimizer.param_groups[0]['lr'])

            # Print profiler results every 10 batches
            #if i % 50 == 0:
            #    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

    # Validation
    model.eval()
    val_loss = 0
    val_accuracy = 0
    with torch.no_grad():
        for images, labels in valloader:
            images = images.view(-1, 28 * 28).to(device)
            output = model(images)
            val_loss += criterion(output, labels.to(device)).item()
            val_accuracy += (
                (output.argmax(dim=1) == labels.to(device)).float().mean().item()
            )
    val_loss /= len(valloader)
    val_accuracy /= len(valloader)

    # Update learning rate
    scheduler.step()

    print(
        f"Epoch {epoch + 1}, Val Loss: {val_loss}, Val Accuracy: {val_accuracy}"
    )
