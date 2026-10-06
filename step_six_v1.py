import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
from torchvision.models import resnet18
import numpy as np
from flwr_datasets import FederatedDataset
from reproducibility import seed_everything
from experiment_common import (build_parser, apply_args, nested_buffer_indices, step_budget,
                               make_result, print_table, save_results_csv, plot_loss)

# ------------------------------------------------------------------
# 1. Configuration Setup
# ------------------------------------------------------------------
CONFIG = {
    "num_clients": 10,
    "batch_size": 32,
    "local_epochs": 5,
    "num_rounds": 5,
    "lr": 0.001,
    "seed": 42,
    "dataset_revision": "aadb3af77e9048adbea6b47c21a81e47dd092ae5",
    "device": "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
}

parser = build_parser("CIFAR-100: fase A y fase B repetidas para cada buffer", default_buffers=[0, 5, 10, 15, 20])
args = parser.parse_args()
apply_args(args, CONFIG, parser)

seed_everything(CONFIG["seed"])
print(f"Device in use: {CONFIG['device']}")

# ------------------------------------------------------------------
# 2. Data Preparation & Filtering
# ------------------------------------------------------------------
fds = FederatedDataset(
    dataset="uoft-cs/cifar100",
    revision=CONFIG["dataset_revision"],
    partitioners={"train": CONFIG["num_clients"]},
    seed=CONFIG["seed"],
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

# ------------------------------------------------------------------
# 4. Replay Buffer Simulation Runner
# ------------------------------------------------------------------
def run_experiment(replay_percent):
    replay_fraction = replay_percent / 100
    # Cada variante reinicia la semilla: la fase A es idéntica en todas y la
    # comparación no depende del orden de ejecución.
    seed_everything(CONFIG["seed"])
    print(f"\n========================================================")
    print(f"  RUNNING EXPERIMENT: Replay Buffer = {replay_percent}%")
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
                indices = nested_buffer_indices(len(client_group_a), replay_fraction, CONFIG["seed"], client_id)
                client_replay_buffers[client_id] = client_group_a.select(indices) if indices else None

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

    # Evaluate baseline accuracies after Step 3 (Phase 1)
    acc_a_after_step3 = evaluate(global_model, test_loader_a, CONFIG["device"])
    acc_b_after_step3 = evaluate(global_model, test_loader_b, CONFIG["device"])

    # --- PHASE 2: Training on Group B + Mixed Replay Buffer ---
    print(f"\n--- PHASE 2: TRAINING ON GROUP B (WITH {int(replay_fraction * 100)}% REPLAY BUFFER) ---")
    for r in range(CONFIG["num_rounds"]):
        local_weights = []
        
        for client_id in range(CONFIG["num_clients"]):
            client_partition = fds.load_partition(partition_id=client_id, split="train")
            client_group_b = filter_by_classes(client_partition, group_b_classes)
            
            # Combine Group B data with the client's Group A replay buffer (if any)
            datasets_to_combine = [client_group_b.with_transform(transform_batch)]
            if client_replay_buffers[client_id] is not None:
                datasets_to_combine.append(client_replay_buffers[client_id].with_transform(transform_batch))
            combined_dataset = ConcatDataset(datasets_to_combine)
            
            train_loader = DataLoader(
                combined_dataset, 
                batch_size=CONFIG["batch_size"], 
                shuffle=True, 
                collate_fn=collate_fn
            )
            
            local_model = get_model().to(CONFIG["device"])
            local_model.load_state_dict(global_model.state_dict())
            
            max_steps = step_budget(len(client_group_b), CONFIG["batch_size"], CONFIG["local_epochs"], CONFIG["equal_steps"])
            w = train_local(local_model, train_loader, CONFIG["local_epochs"], CONFIG["lr"], CONFIG["device"], max_steps=max_steps)
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

    # Final accuracies after Step 4 (Phase 2)
    acc_a_after_step4 = evaluate(global_model, test_loader_a, CONFIG["device"])
    acc_b_after_step4 = evaluate(global_model, test_loader_b, CONFIG["device"])

    buffer_total = sum(len(b) for b in client_replay_buffers.values() if b is not None)
    return make_result(replay_percent, acc_a_after_step3 * 100, acc_a_after_step4 * 100, acc_b_after_step4 * 100,
                       buffer_samples=buffer_total, seed=CONFIG["seed"], equal_steps=CONFIG["equal_steps"])

# ------------------------------------------------------------------
# Run every buffer, including 0 % (no replay), which is measured, not assumed.
# ------------------------------------------------------------------
all_results = [run_experiment(percent) for percent in args.buffers]

# Labels come from each result, so the summary can no longer mislabel buffers.
print_table("STEP 6 SUMMARY RESULTS", all_results)
save_results_csv(all_results)
plot_loss(all_results, "group_a_accuracy_loss_all_buffers.png", "Group A Accuracy Loss Across All Methods")
