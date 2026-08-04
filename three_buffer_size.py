import os
import gc
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
from torchvision.models import resnet18
import numpy as np
import matplotlib.pyplot as plt
from flwr_datasets import FederatedDataset

# ------------------------------------------------------------------
# 1. Configuration Setup
# ------------------------------------------------------------------
CONFIG = {
    "num_clients": 10,
    "batch_size": 32,
    "local_epochs": 5,
    "num_rounds": 5,
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

# Prepare Test Loaders for Group A and Group B
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
# 3. Helper Functions
# ------------------------------------------------------------------
def get_model():
    model = resnet18(num_classes=100)
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

# ------------------------------------------------------------------
# 4. Replay Buffer Simulation Runner
# ------------------------------------------------------------------
def run_experiment(replay_fraction):
    print(f"\n========================================================")
    print(f"  RUNNING EXPERIMENT: Replay Buffer = {int(replay_fraction * 100)}%")
    print(f"========================================================")

    global_model = get_model().to(CONFIG["device"])
    client_replay_buffers = {}

    # --- PHASE 1: Training on Group A & Creating Replay Buffers ---
    print("\n--- PHASE 1: TRAINING ON GROUP A ---")
    for r in range(CONFIG["num_rounds"]):
        local_weights = []
        
        for client_id in range(CONFIG["num_clients"]):
            client_partition = fds.load_partition(partition_id=client_id, split="train")
            client_group_a = filter_by_classes(client_partition, group_a_classes)
            
            # Save a sample of Group A data as Replay Buffer on the last round of Phase 1
            if r == CONFIG["num_rounds"] - 1:
                buffer_size = int(len(client_group_a) * replay_fraction)
                shuffled_indices = np.random.choice(len(client_group_a), size=buffer_size, replace=False)
                client_replay_buffers[client_id] = client_group_a.select(shuffled_indices)

            train_loader = DataLoader(
                client_group_a.with_transform(transform_batch), 
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
        acc_a = evaluate(global_model, test_loader_a, CONFIG["device"])
        print(f"Phase 1 - Round {r+1}/{CONFIG['num_rounds']} -> Group A Accuracy: {acc_a * 100:.2f}%")

    acc_a_after_step3 = evaluate(global_model, test_loader_a, CONFIG["device"])

    # --- PHASE 2: Training on Group B + Mixed Replay Buffer ---
    print(f"\n--- PHASE 2: TRAINING ON GROUP B (WITH {int(replay_fraction * 100)}% REPLAY BUFFER) ---")
    for r in range(CONFIG["num_rounds"]):
        local_weights = []
        
        for client_id in range(CONFIG["num_clients"]):
            client_partition = fds.load_partition(partition_id=client_id, split="train")
            client_group_b = filter_by_classes(client_partition, group_b_classes)
            
            replay_data = client_replay_buffers[client_id]
            
            combined_dataset = ConcatDataset([
                client_group_b.with_transform(transform_batch),
                replay_data.with_transform(transform_batch)
            ])
            
            train_loader = DataLoader(
                combined_dataset, 
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
        
        acc_a = evaluate(global_model, test_loader_a, CONFIG["device"])
        acc_b = evaluate(global_model, test_loader_b, CONFIG["device"])
        
        print(f"Phase 2 - Round {r+1}/{CONFIG['num_rounds']} -> Group A Accuracy: {acc_a * 100:.2f}% | Group B Accuracy: {acc_b * 100:.2f}%")

    acc_a_after_step4 = evaluate(global_model, test_loader_a, CONFIG["device"])
    acc_b_after_step4 = evaluate(global_model, test_loader_b, CONFIG["device"])

    clear_memory()

    return {
        "fraction": f"{int(replay_fraction * 100)}%",
        "step3_acc_a": acc_a_after_step3 * 100,
        "step4_acc_a": acc_a_after_step4 * 100,
        "step4_acc_b": acc_b_after_step4 * 100,
        "loss_a": (acc_a_after_step3 - acc_a_after_step4) * 100
    }

# ------------------------------------------------------------------
# 5. Execute All Replay Experiments (%5, %15, %25)
# ------------------------------------------------------------------
results_5pct = run_experiment(0.05)
results_15pct = run_experiment(0.15)
results_25pct = run_experiment(0.25)

# Baseline (No Replay) Values from Step 5
baseline_results = {
    "fraction": "No Replay",
    "step3_acc_a": 25.06,
    "step4_acc_a": 0.00,
    "step4_acc_b": 20.66,
    "loss_a": 25.06
}

all_results = [baseline_results, results_5pct, results_15pct, results_25pct]

# ------------------------------------------------------------------
# 6. Summary Table Display
# ------------------------------------------------------------------
print("\n" + "="*90)
print("                    FINAL COMPARISON TABLE (STEP 7)")
print("="*90)
print(f"{'Method':<20} | {'Group A (Before)':<18} | {'Group A (After)':<18} | {'Group B (After)':<18} | {'Accuracy Loss':<15}")
print("-" * 95)
for res in all_results:
    print(f"{res['fraction']:<20} | {res['step3_acc_a']:>17.2f}% | {res['step4_acc_a']:>17.2f}% | {res['step4_acc_b']:>17.2f}% | {res['loss_a']:>14.2f}%")
print("="*90)

# ------------------------------------------------------------------
# 7. Plot & Save Chart
# ------------------------------------------------------------------
methods = [res['fraction'] for res in all_results]
accuracy_losses = [res['loss_a'] for res in all_results]
colors = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6']

plt.figure(figsize=(9, 5.5), dpi=300)
bars = plt.bar(methods, accuracy_losses, color=colors, width=0.45, edgecolor='black', linewidth=1)

plt.title('Group A Accuracy Loss Across All Methods', fontsize=14, fontweight='bold', pad=15)
plt.xlabel('Method', fontsize=12, labelpad=10)
plt.ylabel('Accuracy Loss (%)', fontsize=12, labelpad=10)
plt.ylim(0, 30)
plt.grid(axis='y', linestyle='--', alpha=0.5)

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2.0, yval + 0.7, f'{yval:.2f}%', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig('group_a_accuracy_loss_all_buffers.png', dpi=300)
print("\n[INFO] Chart successfully saved as 'group_a_accuracy_loss_all_buffers.png'")
plt.show()