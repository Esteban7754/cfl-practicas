import os
import gc
import copy
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, ConcatDataset, Subset
from torchvision.models import resnet18
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

CONFIG = {
    # MVTec AD dataset directory path:
    "data_dir": "/Users/abdullah/Downloads/mvtec_anomaly_detection", 
    "num_clients": 10,
    "batch_size": 32,
    "local_epochs": 5,
    "num_rounds": 5,
    "lr": 0.001,
    "img_size": 64,  # Fast ResNet training size
    "num_classes": 15,
    "device": "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
}

print(f"Device in use: {CONFIG['device']}")

# 15 Industrial Categories in MVTec AD
ALL_CATEGORIES = [
    "bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather", # Group A (7)
    "metal_nut", "pill", "screw", "tile", "toothbrush", "transistor", "wood", "zipper" # Group B (8)
]

CATEGORY_TO_IDX = {cat: idx for idx, cat in enumerate(ALL_CATEGORIES)}

# Instructor's exact 2-Group Industrial Task Split (7 vs 8 categories)
group_a_classes = [CATEGORY_TO_IDX[c] for c in ["bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather"]]
group_b_classes = [CATEGORY_TO_IDX[c] for c in ["metal_nut", "pill", "screw", "tile", "toothbrush", "transistor", "wood", "zipper"]]

class MVTecDataset(Dataset):
    """
    Custom PyTorch Dataset for MVTec AD.
    Scans image files from the local directory for specified category classes.
    Converts all images to RGB (handles both color and grayscale textures).
    """
    def __init__(self, root_dir, allowed_classes, is_train=True, transform=None):
        self.samples = []
        self.transform = transform
        
        target_split = "train" if is_train else "test"
        
        for class_idx in allowed_classes:
            cat_name = ALL_CATEGORIES[class_idx]
            cat_path = os.path.join(root_dir, cat_name, target_split)
            
            if not os.path.exists(cat_path):
                continue
                
            # Scan all subfolders (e.g. good, defect types)
            image_paths = glob.glob(os.path.join(cat_path, "*", "*.png")) + \
                          glob.glob(os.path.join(cat_path, "*", "*.jpg")) + \
                          glob.glob(os.path.join(cat_path, "*", "*.JPEG"))
                          
            for img_p in image_paths:
                self.samples.append((img_p, class_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        
        if self.transform:
            img = self.transform(img)
            
        return img, label

transform = transforms.Compose([
    transforms.Resize((CONFIG["img_size"], CONFIG["img_size"])),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def get_model():
    """
    Returns a ResNet-18 adapted for 64x64 images and 15 MVTec AD classes.
    """
    model = resnet18(num_classes=CONFIG["num_classes"])
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model

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

def clear_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def create_client_partitions(dataset, num_clients):
    data_len = len(dataset)
    indices = list(range(data_len))
    np.random.shuffle(indices)
    split_size = data_len // num_clients

    client_datasets = []
    for i in range(num_clients):
        client_indices = indices[i * split_size : (i + 1) * split_size]
        client_datasets.append(Subset(dataset, client_indices))
    return client_datasets

def get_root_dataset_and_idx(dataset_obj, idx):
    """
    İç içe geçmiş PyTorch Subset nesnelerini soyup kök MVTecDataset
    nesnesine ve gerçek örnek indeksine ulaşır.
    """
    curr = dataset_obj
    curr_idx = idx
    while isinstance(curr, Subset):
        curr_idx = curr.indices[curr_idx]
        curr = curr.dataset
    return curr, curr_idx

def calculate_avg_client_storage(buffer_dict):
    """
    EXTRA E: Calculates average additional storage cost per client in MB.
    Returns:
      - Disk Storage (MB): Actual image PNG/JPG file size on disk.
      - Tensor Storage (MB): Float32 tensor memory (3 x H x W x 4 bytes + label).
    """
    if not buffer_dict:
        return 0.0, 0.0
        
    disk_mb_list = []
    tensor_mb_list = []
    
    for client_id, subset in buffer_dict.items():
        if subset is None or len(subset) == 0:
            disk_mb_list.append(0.0)
            tensor_mb_list.append(0.0)
            continue
            
        total_disk_bytes = 0
        total_tensor_bytes = 0
        
        for idx in range(len(subset)):
            root_ds, real_idx = get_root_dataset_and_idx(subset, idx)
            img_path, _ = root_ds.samples[real_idx]
            
            # Disk file size
            total_disk_bytes += os.path.getsize(img_path)
            
            # Tensor memory size: 3 channels * 64 * 64 * 4 bytes + 8 bytes (label)
            total_tensor_bytes += (3 * CONFIG["img_size"] * CONFIG["img_size"] * 4) + 8
            
        disk_mb_list.append(total_disk_bytes / (1024 * 1024))
        tensor_mb_list.append(total_tensor_bytes / (1024 * 1024))
        
    return float(np.mean(disk_mb_list)), float(np.mean(tensor_mb_list))

def print_table(title, results_list):
    print("\n" + "="*120)
    print(f"                    {title}")
    print("="*120)
    print(f"{'Method':<12} | {'Group A (Before)':<18} | {'Group A (After)':<18} | {'Group B (After)':<18} | {'Group A Loss':<12} | {'Storage / Client (Disk / Tensor)':<30}")
    print("-" * 120)
    for res in results_list:
        storage_str = f"{res['disk_mb']:.2f} MB / {res['tensor_mb']:.2f} MB"
        print(f"{res['fraction']:<12} | {res['p1_acc_a']:>17.2f}% | {res['p2_acc_a']:>17.2f}% | {res['p2_acc_b']:>17.2f}% | {res['loss_a']:>11.2f}% | {storage_str:>30}")
    print("="*120)

def plot_and_save_chart(results_list, filename, title_text):
    methods = [res['fraction'] for res in results_list]
    loss_a = [res['loss_a'] for res in results_list]
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6'][:len(results_list)]

    plt.figure(figsize=(9, 5.5), dpi=300)
    bars = plt.bar(methods, loss_a, color=colors, width=0.45, edgecolor='black', linewidth=1)

    plt.title(title_text, fontsize=13, fontweight='bold', pad=15)
    plt.xlabel('Replay Buffer Size', fontsize=11, labelpad=10)
    plt.ylabel('Group A Accuracy Loss (%)', fontsize=11, labelpad=10)
    plt.ylim(0, max(loss_a) + 15 if len(loss_a) > 0 and max(loss_a) > 0 else 50)
    plt.grid(axis='y', linestyle='--', alpha=0.5)

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, yval + 0.8, f'{yval:.2f}%', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    print(f"\n[INFO] Chart saved as '{filename}'")
    try:
        plt.show()
    except Exception:
        pass

print(f"\n[INFO] Loading MVTec AD dataset from '{CONFIG['data_dir']}'...")

if not os.path.exists(CONFIG["data_dir"]):
    print(f"\n[WARNING] '{CONFIG['data_dir']}' directory not found!")
    print("Please set CONFIG['data_dir'] to the extracted MVTec AD folder path.")
    exit(1)

# Training Subsets
train_group_a = MVTecDataset(CONFIG["data_dir"], group_a_classes, is_train=True, transform=transform)
train_group_b = MVTecDataset(CONFIG["data_dir"], group_b_classes, is_train=True, transform=transform)

# Testing Loaders
test_group_a = MVTecDataset(CONFIG["data_dir"], group_a_classes, is_train=False, transform=transform)
test_group_b = MVTecDataset(CONFIG["data_dir"], group_b_classes, is_train=False, transform=transform)

test_loader_a = DataLoader(test_group_a, batch_size=CONFIG["batch_size"], shuffle=False)
test_loader_b = DataLoader(test_group_b, batch_size=CONFIG["batch_size"], shuffle=False)

client_a_partitions = create_client_partitions(train_group_a, CONFIG["num_clients"])
client_b_partitions = create_client_partitions(train_group_b, CONFIG["num_clients"])

print(f"[INFO] Train samples - Group A (7 categories): {len(train_group_a)}, Group B (8 categories): {len(train_group_b)}")

print("\n========================================================")
print("  PHASE 1: TRAINING ON GROUP A (BOTTLE, CABLE, CAPSULE, CARPET, GRID, HAZELNUT, LEATHER)")
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

    # FedAvg
    avg_weights = {}
    for key in local_weights[0].keys():
        target_dtype = local_weights[0][key].dtype
        stacked_weights = torch.stack([w[key].to(torch.float32) for w in local_weights], dim=0)
        avg_weights[key] = stacked_weights.mean(dim=0).to(target_dtype)

    global_model.load_state_dict(avg_weights)
    acc_a = evaluate(global_model, test_loader_a, CONFIG["device"])
    print(f"Phase 1 - Round {r+1}/{CONFIG['num_rounds']} -> Group A Accuracy: {acc_a * 100:.2f}%")

base_phase1_weights = copy.deepcopy(global_model.state_dict())
base_acc_a = evaluate(global_model, test_loader_a, CONFIG["device"]) * 100

def run_continual_experiment(replay_fraction):
    print(f"\n========================================================")
    print(f"  RUNNING CONTINUAL EXPERIMENT: Replay Buffer = {int(replay_fraction * 100)}%")
    print(f"========================================================")

    global_model = get_model().to(CONFIG["device"])
    global_model.load_state_dict(copy.deepcopy(base_phase1_weights))

    buffer_a = {}

    # Sample Buffer A from Phase 1
    if replay_fraction > 0:
        for client_id in range(CONFIG["num_clients"]):
            client_data_a = client_a_partitions[client_id]
            size_a = int(len(client_data_a) * replay_fraction)
            if size_a > 0:
                idx_a = np.random.choice(len(client_data_a), size=size_a, replace=False)
                buffer_a[client_id] = Subset(client_data_a, idx_a)

    # EXTRA E: Measure average storage cost per client (Disk MB and Tensor MB)
    disk_mb, tensor_mb = calculate_avg_client_storage(buffer_a)

    # Phase 2: Training on Group B (8 Remaining Categories)
    for r in range(CONFIG["num_rounds"]):
        local_weights = []

        for client_id in range(CONFIG["num_clients"]):
            client_data_b = client_b_partitions[client_id]
            rep_a = buffer_a.get(client_id, None)

            train_ds = ConcatDataset([client_data_b, rep_a]) if rep_a else client_data_b
            train_loader = DataLoader(train_ds, batch_size=CONFIG["batch_size"], shuffle=True)

            local_model = get_model().to(CONFIG["device"])
            local_model.load_state_dict(global_model.state_dict())

            w = train_local(local_model, train_loader, CONFIG["local_epochs"], CONFIG["lr"], CONFIG["device"])
            local_weights.append(w)

        # FedAvg
        avg_weights = {}
        for key in local_weights[0].keys():
            target_dtype = local_weights[0][key].dtype
            stacked_weights = torch.stack([w[key].to(torch.float32) for w in local_weights], dim=0)
            avg_weights[key] = stacked_weights.mean(dim=0).to(target_dtype)
        global_model.load_state_dict(avg_weights)

        acc_a_curr = evaluate(global_model, test_loader_a, CONFIG["device"]) * 100
        acc_b_curr = evaluate(global_model, test_loader_b, CONFIG["device"]) * 100
        print(f"Phase 2 - Round {r+1}/{CONFIG['num_rounds']} -> Group A Acc: {acc_a_curr:.2f}% | Group B Acc: {acc_b_curr:.2f}%")

    acc_a_end = evaluate(global_model, test_loader_a, CONFIG["device"]) * 100
    acc_b_end = evaluate(global_model, test_loader_b, CONFIG["device"]) * 100

    clear_memory()

    return {
        "fraction": f"{int(replay_fraction * 100)}%" if replay_fraction > 0 else "No Replay",
        "p1_acc_a": base_acc_a,
        "p2_acc_a": acc_a_end,
        "p2_acc_b": acc_b_end,
        "loss_a": base_acc_a - acc_a_end,
        "disk_mb": disk_mb,
        "tensor_mb": tensor_mb
    }

results_0pct = run_continual_experiment(0.00)
results_5pct = run_continual_experiment(0.05)
results_15pct = run_continual_experiment(0.15)

results_stage_1 = [results_0pct, results_5pct, results_15pct]
print_table("INTERMEDIATE TABLE 1 (MVTEC AD - AFTER 5% & 15% REPLAY)", results_stage_1)
plot_and_save_chart(results_stage_1, "mvtec_continual_loss_5_15.png", "MVTec AD Group A Accuracy Loss (No Replay vs 5% & 15%)")

results_25pct = run_continual_experiment(0.25)

results_stage_2 = [results_0pct, results_5pct, results_15pct, results_25pct]
print_table("FINAL COMPARISON TABLE 2 (MVTEC AD - ALL RESULTS WITH EXTRA E)", results_stage_2)
plot_and_save_chart(results_stage_2, "mvtec_continual_loss_all.png", "MVTec AD Group A Accuracy Loss Across All Buffer Sizes")