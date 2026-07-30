import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.models import resnet18
import numpy as np
from flwr_datasets import FederatedDataset

# ------------------------------------------------------------------
# 1. Configuration Setup
# ------------------------------------------------------------------
CONFIG = {
    "num_clients": 10,
    "batch_size": 32,
    "local_epochs": 1,    # Number of local training epochs per client per round
    "num_rounds": 5,      # Total number of Federated Learning rounds
    "lr": 0.001,
    "device": "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
}

print(f"Device in use: {CONFIG['device']}")

# ------------------------------------------------------------------
# 2. Data Preparation & Filtering
# ------------------------------------------------------------------
fds = FederatedDataset(
    dataset="uoft-cs/cifar100",
    partitioners={"train": CONFIG["num_clients"]},
)

group_a_classes = list(range(0, 50))
group_b_classes = list(range(50, 100))

def filter_by_classes(partition, class_labels):
    allowed_classes = set(class_labels)
    label_col = "fine_label" if "fine_label" in partition.column_names else "label"
    return partition.filter(lambda example: example[label_col] in allowed_classes)

def transform_batch(batch):
    # Convert HuggingFace PIL images to PyTorch Tensors (3, 32, 32) and normalize to [0, 1]
    images = [
        torch.tensor(np.array(img), dtype=torch.float32).permute(2, 0, 1) / 255.0 
        for img in batch["img"]
    ]
    labels = torch.tensor(batch["fine_label"] if "fine_label" in batch else batch["label"], dtype=torch.long)
    return {"img": images, "label": labels}

# Prepare Test Set for Group A evaluation
test_dataset = fds.load_split("test")
test_group_a = filter_by_classes(test_dataset, group_a_classes)

def collate_fn(batch):
    imgs = torch.stack([x["img"] for x in batch])
    labels = torch.tensor([x["label"] for x in batch])
    return imgs, labels

test_loader = DataLoader(
    test_group_a.with_transform(transform_batch), 
    batch_size=CONFIG["batch_size"], 
    collate_fn=collate_fn
)

# ------------------------------------------------------------------
# 3. Model Architecture (ResNet-18, num_classes=100)
# ------------------------------------------------------------------
def get_model():
    model = resnet18(num_classes=100)
    # Adapt the first conv layer for 32x32 CIFAR-100 images
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model

# ------------------------------------------------------------------
# 4. Training and Evaluation Functions
# ------------------------------------------------------------------
def train_local(model, train_loader, epochs, lr, device):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    for _ in range(epochs):
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
    return model.state_dict()

def evaluate(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total if total > 0 else 0.0

# ------------------------------------------------------------------
# 5. Simulated FedAvg Process (Phase 1: Group A Only)
# ------------------------------------------------------------------
global_model = get_model().to(CONFIG["device"])

print("\n--- PHASE 1: TRAINING ON GROUP A STARTED ---")
group_a_accuracies = []

for r in range(CONFIG["num_rounds"]):
    local_weights = []
    
    # Perform local training on each client using Group A data
    for client_id in range(CONFIG["num_clients"]):
        client_partition = fds.load_partition(partition_id=client_id, split="train")
        client_group_a = filter_by_classes(client_partition, group_a_classes)
        
        train_loader = DataLoader(
            client_group_a.with_transform(transform_batch), 
            batch_size=CONFIG["batch_size"], 
            shuffle=True, 
            collate_fn=collate_fn
        )
        
        # Copy global model weights to client
        local_model = get_model().to(CONFIG["device"])
        local_model.load_state_dict(global_model.state_dict())
        
        # Train locally
        w = train_local(local_model, train_loader, CONFIG["local_epochs"], CONFIG["lr"], CONFIG["device"])
        local_weights.append(w)
    
    # FedAvg: Weight Aggregation
    avg_weights = {}
    for key in local_weights[0].keys():
        # Save original data type (e.g., Long for num_batches_tracked)
        target_dtype = local_weights[0][key].dtype

        # Cast tensors to float32 before computing the mean
        stacked_weights = torch.stack([w[key].to(torch.float32) for w in local_weights], dim=0)
        avg_tensor = stacked_weights.mean(dim=0)

        # Convert back to target data type
        avg_weights[key] = avg_tensor.to(target_dtype)

    global_model.load_state_dict(avg_weights)
    
    # Evaluate Group A performance at the end of each round
    acc = evaluate(global_model, test_loader, CONFIG["device"])
    group_a_accuracies.append(acc)
    print(f"Round {r+1}/{CONFIG['num_rounds']} - Group A Accuracy: {acc * 100:.2f}%")

# Save Baseline Model
torch.save(global_model.state_dict(), "global_model_phase1.pt")
print("\nPhase 1 Completed! Baseline model saved to 'global_model_phase1.pt'.")