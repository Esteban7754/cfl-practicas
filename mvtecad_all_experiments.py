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
from reproducibility import seed_everything
from experiment_common import (build_parser, apply_args, nested_buffer_indices, step_budget,
                               make_result, print_table, save_results_csv, plot_loss)

CONFIG = {
    # Dataset root stored beside this script for portable, repeatable runs.
    "data_dir": os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "mvtec_anomaly_detection"),
    "num_clients": 10,
    "batch_size": 32,
    "local_epochs": 5,
    "num_rounds": 5,
    "lr": 0.001,
    "img_size": 64,  # Fast ResNet training size
    "num_classes": 15,
    "seed": 42,
    "device": "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
}

parser = build_parser("MVTec AD: 7 categorías (A) y luego 8 (B) con distintos buffers")
args = parser.parse_args()
if not os.path.exists(CONFIG["data_dir"]):
    print(f"\n[WARNING] '{CONFIG['data_dir']}' directory not found!")
    print("Extract the MVTec AD dataset so its 15 category folders are inside this directory.")
    raise SystemExit(1)
apply_args(args, CONFIG, parser)

seed_everything(CONFIG["seed"])
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

def train_local(model, train_loader, epochs, lr, device, max_steps=None):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    steps = 0
    for _ in range(epochs):
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            steps += 1
            if max_steps is not None and steps >= max_steps:
                return model.state_dict()

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

print(f"\n[INFO] Loading MVTec AD dataset from '{CONFIG['data_dir']}'...")


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

def run_continual_experiment(replay_percent):
    replay_fraction = replay_percent / 100
    seed_everything(CONFIG["seed"])  # misma semilla en todas las variantes
    print(f"\n========================================================")
    print(f"  RUNNING CONTINUAL EXPERIMENT: Replay Buffer = {replay_percent}%")
    print(f"========================================================")

    global_model = get_model().to(CONFIG["device"])
    global_model.load_state_dict(copy.deepcopy(base_phase1_weights))

    buffer_a = {}

    # Sample Buffer A from Phase 1
    if replay_fraction > 0:
        for client_id in range(CONFIG["num_clients"]):
            client_data_a = client_a_partitions[client_id]
            idx_a = nested_buffer_indices(len(client_data_a), replay_fraction, CONFIG["seed"], client_id)
            if idx_a:
                buffer_a[client_id] = Subset(client_data_a, idx_a)

    # EXTRA E: Measure average storage cost per client (Disk MB and Tensor MB)
    disk_mb, tensor_mb = calculate_avg_client_storage(buffer_a)

    # Phase 2: Training on Group B (8 Remaining Categories)
    for r in range(CONFIG["num_rounds"]):
        local_weights = []

        for client_id in range(CONFIG["num_clients"]):
            client_data_b = client_b_partitions[client_id]
            rep_a = buffer_a.get(client_id, None)

            train_ds = ConcatDataset([client_data_b, rep_a]) if rep_a is not None else client_data_b
            train_loader = DataLoader(train_ds, batch_size=CONFIG["batch_size"], shuffle=True)

            local_model = get_model().to(CONFIG["device"])
            local_model.load_state_dict(global_model.state_dict())

            max_steps = step_budget(len(client_data_b), CONFIG["batch_size"], CONFIG["local_epochs"], CONFIG["equal_steps"])
            w = train_local(local_model, train_loader, CONFIG["local_epochs"], CONFIG["lr"], CONFIG["device"], max_steps=max_steps)
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

    buffer_total = sum(len(b) for b in buffer_a.values())
    return make_result(replay_percent, base_acc_a, acc_a_end, acc_b_end,
                       buffer_samples=buffer_total, disk_mb=disk_mb, tensor_mb=tensor_mb,
                       seed=CONFIG["seed"], equal_steps=CONFIG["equal_steps"])

all_results = [run_continual_experiment(percent) for percent in args.buffers]
up_to_20 = [r for r in all_results if r["replay_percent"] <= 20]

print_table("FINAL COMPARISON TABLE (MVTEC AD)", all_results)
print("Coste medio de almacenamiento por cliente (disco / tensores):")
for r in all_results:
    print(f"  {r['fraction']:<10} {r['disk_mb']:.2f} MB / {r['tensor_mb']:.2f} MB")
save_results_csv(all_results)
plot_loss(up_to_20, "mvtec_continual_loss_up_to_20.png", "MVTec AD Group A Accuracy Loss (buffers up to 20%)", xlabel="Replay Buffer Size")
plot_loss(all_results, "mvtec_continual_loss_all.png", "MVTec AD Group A Accuracy Loss Across All Buffer Sizes", xlabel="Replay Buffer Size")
