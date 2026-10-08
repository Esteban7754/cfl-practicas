import os
import gc
import copy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, ConcatDataset, Subset
import torchvision.transforms as transforms
import numpy as np
from datasets import load_dataset
import fl_core
from reproducibility import seed_everything
from experiment_common import (build_parser, apply_args, nested_buffer_indices, step_budget,
                               make_result, print_table, save_results_csv, plot_loss)

# Configuration Setup
CONFIG = {
    "num_clients": 10,
    "batch_size": 32,
    "local_epochs": 3,
    "num_rounds": 5,
    "lr": 0.001,
    "num_classes": 10,  # Focus on the first 10 distinct classes
    "seed": 42,
    "dataset_revision": "ee20570ae7a29c51571e55a9a17983f7625295d6",
    "device": "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
}

parser = build_parser("DomainNet: fotos reales (A) y luego bocetos (B) con distintos buffers")
args = parser.parse_args()
apply_args(args, CONFIG, parser)

seed_everything(CONFIG["seed"])
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
    """ResNet-18 de fl_core adaptada a imágenes pequeñas, con CONFIG["num_classes"] salidas."""
    return fl_core.get_model(CONFIG["num_classes"])

def train_local(model, train_loader, epochs, lr, device, max_steps=None):
    """Entrenamiento local común (fl_core) sobre un DataLoader."""
    return fl_core.train_local(model, fl_core.loader_batches(train_loader, epochs, max_steps), lr, device)

def evaluate(model, test_loader, device):
    return fl_core.accuracy_on_loader(model, test_loader, device)

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

print("\n[INFO] Loading DomainNet dataset from Hugging Face...")

# NOTE: "greg_h/domainnet" does not exist on the Hub. Using the real, public
# DomainNet mirror instead: wltjr1007/DomainNet. That dataset keeps ALL
# domains together in one split (train/test), distinguished by a `domain`
# column, instead of separate per-domain configs -- so we load it once and
# filter by domain name below.
raw_train = load_dataset("wltjr1007/DomainNet", split="train", revision=CONFIG["dataset_revision"])
raw_test = load_dataset("wltjr1007/DomainNet", split="test", revision=CONFIG["dataset_revision"])

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
    
    global_model.load_state_dict(fl_core.fedavg(local_weights))
    acc_a = evaluate(global_model, test_loader_a, CONFIG["device"])
    print(f"Phase 1 - Round {r+1}/{CONFIG['num_rounds']} -> Domain A (Real) Accuracy: {acc_a * 100:.2f}%")

base_group_a_weights = copy.deepcopy(global_model.state_dict())
base_acc_a = evaluate(global_model, test_loader_a, CONFIG["device"]) * 100

def run_phase2_experiment(replay_percent):
    replay_fraction = replay_percent / 100
    seed_everything(CONFIG["seed"])  # misma semilla en todas las variantes
    print(f"\n========================================================")
    print(f"  RUNNING PHASE 2 (SKETCH): Replay Buffer = {replay_percent}%")
    print(f"========================================================")

    global_model = get_model().to(CONFIG["device"])
    global_model.load_state_dict(copy.deepcopy(base_group_a_weights))

    client_replay_buffers = {}
    for client_id in range(CONFIG["num_clients"]):
        client_data_a = client_a_partitions[client_id]
        indices = nested_buffer_indices(len(client_data_a), replay_fraction, CONFIG["seed"], client_id)
        client_replay_buffers[client_id] = Subset(client_data_a, indices) if indices else None

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
            
            max_steps = step_budget(len(client_data_b), CONFIG["batch_size"], CONFIG["local_epochs"], CONFIG["equal_steps"])
            w = train_local(local_model, train_loader, CONFIG["local_epochs"], CONFIG["lr"], CONFIG["device"], max_steps=max_steps)
            local_weights.append(w)
        
        global_model.load_state_dict(fl_core.fedavg(local_weights))
        
        acc_a = evaluate(global_model, test_loader_a, CONFIG["device"])
        acc_b = evaluate(global_model, test_loader_b, CONFIG["device"])
        
        print(f"Phase 2 - Round {r+1}/{CONFIG['num_rounds']} -> Domain A (Real) Acc: {acc_a * 100:.2f}% | Domain B (Sketch) Acc: {acc_b * 100:.2f}%")

    acc_a_after = evaluate(global_model, test_loader_a, CONFIG["device"]) * 100
    acc_b_after = evaluate(global_model, test_loader_b, CONFIG["device"]) * 100

    clear_memory()

    buffer_total = sum(len(b) for b in client_replay_buffers.values() if b is not None)
    return make_result(replay_percent, base_acc_a, acc_a_after, acc_b_after,
                       buffer_samples=buffer_total, seed=CONFIG["seed"], equal_steps=CONFIG["equal_steps"])

# Execute replay buffer experiments (0 % = no replay, measured like the rest)
all_results = [run_phase2_experiment(percent) for percent in args.buffers]
up_to_20 = [r for r in all_results if r["replay_percent"] <= 20]

print_table("FINAL COMPARISON TABLE (DOMAINNET)", all_results, group_a="Real", group_b="Sketch")
save_results_csv(all_results)
plot_loss(up_to_20, "domainnet_accuracy_loss_up_to_20.png", "Domain A Accuracy Loss (buffers up to 20%)", xlabel="Replay Buffer Size")
plot_loss(all_results, "domainnet_accuracy_loss_all_buffers.png", "Domain A Accuracy Loss Across All Replay Buffers", xlabel="Replay Buffer Size")
