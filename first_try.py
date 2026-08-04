import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.models import resnet18
import numpy as np
from flwr_datasets import FederatedDataset

# ------------------------------------------------------------------
# 1. Konfigürasyon Yapısı (Dökümanın istediği Kolay Değiştirilebilir Yapı)
# ------------------------------------------------------------------
CONFIG = {
    "num_clients": 10,
    "batch_size": 32,
    "local_epochs": 1,    # Her client'ın tur başına yerel eğitim epoch sayısı
    "num_rounds": 5,      # Federated Learning toplam tur sayısı (Test için 5-10 verebilirsin)
    "lr": 0.001,
    "device": "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
}

print(f"Çalışma Cihazı: {CONFIG['device']}")

# ------------------------------------------------------------------
# 2. Veri Hazırlığı & Filtreleme
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
    # HuggingFace PIL resimlerini PyTorch Tensor formatına (3, 32, 32) ve 0-1 aralığına getirir
    images = [
        torch.tensor(np.array(img), dtype=torch.float32).permute(2, 0, 1) / 255.0 
        for img in batch["img"]
    ]
    labels = torch.tensor(batch["fine_label"] if "fine_label" in batch else batch["label"], dtype=torch.long)
    return {"img": images, "label": labels}

# Test Setini Hazırla (Grup A Başarısını Ölçmek İçin)
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
# 3. Model Tanımı (ResNet-18, num_classes=100)
# ------------------------------------------------------------------
def get_model():
    model = resnet18(num_classes=100)
    # CIFAR-100 görselleri 32x32 olduğu için ResNet'in ilk katmanını CIFAR boyutuna uygun hale getiriyoruz
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model

# ------------------------------------------------------------------
# 4. Eğitim ve Değerlendirme Fonksiyonları
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
# 5. Simulated FedAvg Süreci (Faz 1: Sadece Grup A)
# ------------------------------------------------------------------
global_model = get_model().to(CONFIG["device"])

print("\n--- FAZ 1: GRUP A İLE EĞİTİM BAŞLIYOR ---")
group_a_accuracies = []

for r in range(CONFIG["num_rounds"]):
    local_weights = []
    
    # Her client kendi Grup A verisiyle yerel eğitim yapar
    for client_id in range(CONFIG["num_clients"]):
        client_partition = fds.load_partition(partition_id=client_id, split="train")
        client_group_a = filter_by_classes(client_partition, group_a_classes)
        
        train_loader = DataLoader(
            client_group_a.with_transform(transform_batch), 
            batch_size=CONFIG["batch_size"], 
            shuffle=True, 
            collate_fn=collate_fn
        )
        
        # Global modeli client'a kopyala
        local_model = get_model().to(CONFIG["device"])
        local_model.load_state_dict(global_model.state_dict())
        
        # Yerel eğitim yap
        w = train_local(local_model, train_loader, CONFIG["local_epochs"], CONFIG["lr"], CONFIG["device"])
        local_weights.append(w)
    
    # FedAvg: Ağırlıkları Ortalaması (Aggregation)
    avg_weights = {}
    for key in local_weights[0].keys():
        # 1. İlk client'ın orijinal veri tipini sakla
        target_dtype = local_weights[0][key].dtype

        # 2. Tensor'ları float'a çevirip ortalamasını al
        stacked_weights = torch.stack([w[key].to(torch.float32) for w in local_weights], dim=0)
        avg_tensor = stacked_weights.mean(dim=0)

        # 3. Orijinal veri tipine (Long/Int vb.) geri döndür
        avg_weights[key] = avg_tensor.to(target_dtype)

    
    global_model.load_state_dict(avg_weights)
    
    # Tur sonu Grup A Başarısını Ölç
    acc = evaluate(global_model, test_loader, CONFIG["device"])
    group_a_accuracies.append(acc)
    print(f"Round {r+1}/{CONFIG['num_rounds']} - Group A Accuracy: %{acc * 100:.2f}")

# Eğitilmiş Baseline Modeli Kaydet
torch.save(global_model.state_dict(), "global_model_phase1.pt")
print("\nFaz 1 Tamamlandı! Model 'global_model_phase1.pt' olarak kaydedildi.")