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
    "local_epochs": 1,
    "num_rounds": 5,      # Number of rounds for Phase 2
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
    images = [
        torch.tensor(np.array(img), dtype=torch.float32).permute(2, 0, 1) / 255.0 
        for img in batch["img"]
    ]
    labels = torch.tensor(batch["fine_label"] if "fine_label" in batch else batch["label"], dtype=torch.long)
    return {"img": images, "label": labels}

def collate_fn(batch):
    imgs = torch.stack([x["img"] for x in batch])
    labels = torch.tensor([x["label"] for x in batch])
    return imgs, labels

# Prepare Test Sets for both Group A and Group B
test_dataset = fds.load_split("test")

test_group_a = filter_by_classes(test_dataset, group_a_classes)
test_group_b = filter_by_classes(test_dataset, group_b_classes)

test_loader_a = DataLoader(
    test_group_a.with_transform(transform_batch), 
    batch_size=CONFIG["batch_size"], 
    collate_fn=collate_fn
)

test_loader_b = DataLoader(
    test_group_b.with_transform(transform_batch), 
    batch_size=CONFIG["batch_size"], 
    collate_fn=collate_fn
)

# ------------------------------------------------------------------
# 3. Model Architecture & Loading Phase 1 Checkpoint
# ------------------------------------------------------------------
def get_model():
    model = resnet18(num_classes=100)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model

global_model = get_model().to(CONFIG["device"])

# Load pre-trained weights from Phase 1
global_model.load_state_dict(torch.load("global_model_phase1.pt", map_location=CONFIG["device"]))
print("Successfully loaded 'global_model_phase1.pt'.")

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
# 5. Simulated FedAvg Process (Phase 2: Distribution Shift to Group B)
# ------------------------------------------------------------------
print("\n--- PHASE 2: TRAINING ON GROUP B STARTED (DISTRIBUTION SHIFT) ---")

# Initial evaluation before Phase 2 training
initial_acc_a = evaluate(global_model, test_loader_a, CONFIG["device"])
initial_acc_b = evaluate(global_model, test_loader_b, CONFIG["device"])
print(f"Initial Baseline -> Group A Acc: {initial_acc_a * 100:.2f}%, Group B Acc: {initial_acc_b * 100:.2f}%\n")

for r in range(CONFIG["num_rounds"]):
    local_weights = []
    
    # Each client trains ONLY on Group B data
    for client_id in range(CONFIG["num_clients"]):
        client_partition = fds.load_partition(partition_id=client_id, split="train")
        client_group_b = filter_by_classes(client_partition, group_b_classes)
        
        train_loader = DataLoader(
            client_group_b.with_transform(transform_batch), 
            batch_size=CONFIG["batch_size"], 
            shuffle=True, 
            collate_fn=collate_fn
        )
        
        local_model = get_model().to(CONFIG["device"])
        local_model.load_state_dict(global_model.state_dict())
        
        w = train_local(local_model, train_loader, CONFIG["local_epochs"], CONFIG["lr"], CONFIG["device"])
        local_weights.append(w)
    
    # FedAvg Aggregation
    avg_weights = {}
    for key in local_weights[0].keys():
        target_dtype = local_weights[0][key].dtype
        stacked_weights = torch.stack([w[key].to(torch.float32) for w in local_weights], dim=0)
        avg_tensor = stacked_weights.mean(dim=0)
        avg_weights[key] = avg_tensor.to(target_dtype)

    global_model.load_state_dict(avg_weights)
    
    # Evaluate performance on BOTH Group A and Group B test sets
    acc_a = evaluate(global_model, test_loader_a, CONFIG["device"])
    acc_b = evaluate(global_model, test_loader_b, CONFIG["device"])
    
    print(f"Round {r+1}/{CONFIG['num_rounds']} -> Group A Acc (Forgotten): {acc_a * 100:.2f}% | Group B Acc (New): {acc_b * 100:.2f}%")

# Save Phase 2 Model
torch.save(global_model.state_dict(), "global_model_phase2.pt")
print("\nPhase 2 Completed! Shifted model saved to 'global_model_phase2.pt'.")