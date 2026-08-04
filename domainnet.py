import os
import gc
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, ConcatDataset, Subset
from torchvision.models import resnet18
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from datasets import load_dataset

# Configuration Setup
CONFIG = {
    "num_clients": 10,
    "batch_size": 32,
    "local_epochs": 3,
    "num_rounds": 5,
    "lr": 0.001,
    "num_classes": 10,  # Focus on the first 10 distinct classes
    "device": "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
}

print(f"Device in use: {CONFIG['device']}")

class DomainNetDataset(Dataset):
    """
    Custom PyTorch Dataset for DomainNet.
    Filters the first N classes (label < num_classes) without modulo operations
    to preserve distinct class semantic labels. Optionally also filters by a
    single domain (since wltjr1007/DomainNet stores all domains together in
    one split, distinguished by a `domain` column).
    """
    def __init__(self, hf_ds, transform=None, num_classes=10, domain_idx=None):
        self.samples = []
        self.transform = transform

        # Filter items belonging to the first N classes (and, if given, one domain)
        for item in hf_ds:
            if domain_idx is not None and item["domain"] != domain_idx:
                continue
            label = item["label"]
            if label < num_classes:
                img = item["image"].convert("RGB") if hasattr(item["image"], "convert") else item["image"]
                self.samples.append((img, label))
            
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        img, label = self.samples[idx]
        if self.transform:
            img = self.transform(img)
        return img, label

# Image Transforms for DomainNet (Resized to 64x64 for efficient ResNet training)
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def get_model():
    """
    Returns a ResNet-18 modified for 64x64 inputs with 10 output classes.
    """
    model = resnet18(num_classes=CONFIG["num_classes"])
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model

def train_local(model, train_loader, epochs, lr, device):
    """
    Trains the local model on client dataset for specified local epochs.
    """
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
    """
    Evaluates global model accuracy on a target test set.
    """
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

def clear_memory():
    """
    Frees unused memory cache on MPS / CUDA.
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def create_client_partitions(dataset, num_clients):
    """
    Splits dataset into equal non-overlapping subsets for federated clients.
    """
    data_len = len(dataset)
    indices = list(range(data_len))
    np.random.shuffle(indices)
    split_size = data_len // num_clients
    
    client_datasets = []
    for i in range(num_clients):
        client_indices = indices[i * split_size : (i + 1) * split_size]
        client_datasets.append(Subset(dataset, client_indices))
    return client_datasets

def print_table(title, results_list):
    """
    Prints a formatted summary table of accuracy and loss values.
    """
    print("\n" + "="*95)
    print(f"                    {title}")
    print("="*95)
    print(f"{'Method':<20} | {'Domain A (Before)':<18} | {'Domain A (After)':<18} | {'Domain B (After)':<18} | {'Accuracy Loss':<15}")
    print("-" * 100)
    for res in results_list:
        print(f"{res['fraction']:<20} | {res['step3_acc_a']:>17.2f}% | {res['step4_acc_a']:>17.2f}% | {res['step4_acc_b']:>17.2f}% | {res['loss_a']:>14.2f}%")
    print("="*95)

def plot_and_save_chart(results_list, filename, title_text):
    """
    Plots and saves a high-DPI bar chart showing Group A accuracy loss.
    """
    methods = [res['fraction'] for res in results_list]
    accuracy_losses = [res['loss_a'] for res in results_list]
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6'][:len(results_list)]

    plt.figure(figsize=(9, 5.5), dpi=300)
    bars = plt.bar(methods, accuracy_losses, color=colors, width=0.45, edgecolor='black', linewidth=1)

    plt.title(title_text, fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Replay Buffer Size', fontsize=12, labelpad=10)
    plt.ylabel('Domain A Accuracy Loss (%)', fontsize=12, labelpad=10)
    plt.ylim(0, max(accuracy_losses) + 10 if len(accuracy_losses) > 0 and max(accuracy_losses) > 0 else 50)
    plt.grid(axis='y', linestyle='--', alpha=0.5)

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, yval + 0.7, f'{yval:.2f}%', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    print(f"\n[INFO] Chart successfully saved as '{filename}'")
    plt.show()

print("\n[INFO] Loading DomainNet dataset from Hugging Face...")

# NOTE: "greg_h/domainnet" does not exist on the Hub. Using the real, public
# DomainNet mirror instead: wltjr1007/DomainNet. That dataset keeps ALL
# domains together in one split (train/test), distinguished by a `domain`
# column, instead of separate per-domain configs -- so we load it once and
# filter by domain name below.
raw_train = load_dataset("wltjr1007/DomainNet", split="train")
raw_test = load_dataset("wltjr1007/DomainNet", split="test")

# "domain" is a class_label column; map domain name -> integer index
# (don't hardcode the index, the ordering isn't guaranteed to be stable).
domain_names = raw_train.features["domain"].names
print(f"[INFO] Available domains: {domain_names}")
real_idx = domain_names.index("real")
sketch_idx = domain_names.index("sketch")

ds_real_train = raw_train
ds_real_test = raw_test
ds_sketch_train = raw_train
ds_sketch_test = raw_test

print("\n[INFO] Transforming images into PyTorch Datasets (Filtering first 10 classes)...")
dataset_real_train = DomainNetDataset(ds_real_train, transform=transform, num_classes=CONFIG["num_classes"], domain_idx=real_idx)
dataset_real_test = DomainNetDataset(ds_real_test, transform=transform, num_classes=CONFIG["num_classes"], domain_idx=real_idx)

dataset_sketch_train = DomainNetDataset(ds_sketch_train, transform=transform, num_classes=CONFIG["num_classes"], domain_idx=sketch_idx)
dataset_sketch_test = DomainNetDataset(ds_sketch_test, transform=transform, num_classes=CONFIG["num_classes"], domain_idx=sketch_idx)

print(f"[INFO] real train/test sizes: {len(dataset_real_train)} / {len(dataset_real_test)}")
print(f"[INFO] sketch train/test sizes: {len(dataset_sketch_train)} / {len(dataset_sketch_test)}")

test_loader_a = DataLoader(dataset_real_test, batch_size=CONFIG["batch_size"], shuffle=False)
test_loader_b = DataLoader(dataset_sketch_test, batch_size=CONFIG["batch_size"], shuffle=False)

client_a_partitions = create_client_partitions(dataset_real_train, CONFIG["num_clients"])
client_b_partitions = create_client_partitions(dataset_sketch_train, CONFIG["num_clients"])

print("\n========================================================")
print("  PHASE 1: TRAINING ON DOMAIN A (REAL PHOTOS) - ONCE FOR ALL")
print("========================================================")

global_model = get_model().to(CONFIG["device"])

for r in range(CONFIG["num_rounds"]):
    local_weights = []
    
    for client_id in range(CONFIG["num_clients"]):
        client_data_a = client_a_partitions[client_id]
        train_loader = DataLoader(client_data_a, batch_size=CONFIG["batch_size"], shuffle=True)
        
        local_model = get_model().to(CONFIG["device"])
        local_model.load_state_dict(global_model.state_dict())
        
        w = train_local(local_model, train_loader, CONFIG["local_epochs"], CONFIG["lr"], CONFIG["device"])
        local_weights.append(w)
    
    # FedAvg Aggregation
    avg_weights = {}
    for key in local_weights[0].keys():
        target_dtype = local_weights[0][key].dtype
        stacked_weights = torch.stack([w[key].to(torch.float32) for w in local_weights], dim=0)
        avg_weights[key] = stacked_weights.mean(dim=0).to(target_dtype)

    global_model.load_state_dict(avg_weights)
    acc_a = evaluate(global_model, test_loader_a, CONFIG["device"])
    print(f"Phase 1 - Round {r+1}/{CONFIG['num_rounds']} -> Domain A (Real) Accuracy: {acc_a * 100:.2f}%")

base_group_a_weights = copy.deepcopy(global_model.state_dict())
base_acc_a = evaluate(global_model, test_loader_a, CONFIG["device"]) * 100

def run_phase2_experiment(replay_fraction):
    print(f"\n========================================================")
    print(f"  RUNNING PHASE 2 (SKETCH): Replay Buffer = {int(replay_fraction * 100)}%")
    print(f"========================================================")

    global_model = get_model().to(CONFIG["device"])
    global_model.load_state_dict(copy.deepcopy(base_group_a_weights))

    client_replay_buffers = {}
    for client_id in range(CONFIG["num_clients"]):
        client_data_a = client_a_partitions[client_id]
        buffer_size = int(len(client_data_a) * replay_fraction)
        if buffer_size > 0:
            shuffled_indices = np.random.choice(len(client_data_a), size=buffer_size, replace=False)
            client_replay_buffers[client_id] = Subset(client_data_a, shuffled_indices)
        else:
            client_replay_buffers[client_id] = None

    for r in range(CONFIG["num_rounds"]):
        local_weights = []
        
        for client_id in range(CONFIG["num_clients"]):
            client_data_b = client_b_partitions[client_id]
            datasets_to_combine = [client_data_b]
            
            if client_replay_buffers[client_id] is not None:
                datasets_to_combine.append(client_replay_buffers[client_id])
            
            combined_dataset = ConcatDataset(datasets_to_combine)
            train_loader = DataLoader(combined_dataset, batch_size=CONFIG["batch_size"], shuffle=True)
            
            local_model = get_model().to(CONFIG["device"])
            local_model.load_state_dict(global_model.state_dict())
            
            w = train_local(local_model, train_loader, CONFIG["local_epochs"], CONFIG["lr"], CONFIG["device"])
            local_weights.append(w)
        
        # FedAvg Aggregation
        avg_weights = {}
        for key in local_weights[0].keys():
            target_dtype = local_weights[0][key].dtype
            stacked_weights = torch.stack([w[key].to(torch.float32) for w in local_weights], dim=0)
            avg_weights[key] = stacked_weights.mean(dim=0).to(target_dtype)

        global_model.load_state_dict(avg_weights)
        
        acc_a = evaluate(global_model, test_loader_a, CONFIG["device"])
        acc_b = evaluate(global_model, test_loader_b, CONFIG["device"])
        
        print(f"Phase 2 - Round {r+1}/{CONFIG['num_rounds']} -> Domain A (Real) Acc: {acc_a * 100:.2f}% | Domain B (Sketch) Acc: {acc_b * 100:.2f}%")

    acc_a_after = evaluate(global_model, test_loader_a, CONFIG["device"]) * 100
    acc_b_after = evaluate(global_model, test_loader_b, CONFIG["device"]) * 100

    clear_memory()

    return {
        "fraction": f"{int(replay_fraction * 100)}%",
        "step3_acc_a": base_acc_a,
        "step4_acc_a": acc_a_after,
        "step4_acc_b": acc_b_after,
        "loss_a": base_acc_a - acc_a_after
    }

# Execute 5% and 15% Replay Experiments
results_5pct = run_phase2_experiment(0.05)
results_15pct = run_phase2_experiment(0.15)
results_0pct = run_phase2_experiment(0.00)  # No Replay Baseline

results_stage_1 = [results_0pct, results_5pct, results_15pct]
print_table("INTERMEDIATE TABLE (AFTER 5% & 15% REPLAY)", results_stage_1)
plot_and_save_chart(results_stage_1, "domainnet_accuracy_loss_5_15.png", "Domain A Accuracy Loss (No Replay vs 5% & 15%)")

results_25pct = run_phase2_experiment(0.25)

results_stage_2 = [results_0pct, results_5pct, results_15pct, results_25pct]
print_table("FINAL COMPARISON TABLE (AFTER 25% REPLAY - ALL RESULTS)", results_stage_2)
plot_and_save_chart(results_stage_2, "domainnet_accuracy_loss_all_buffers.png", "Domain A Accuracy Loss Across All Replay Buffers")