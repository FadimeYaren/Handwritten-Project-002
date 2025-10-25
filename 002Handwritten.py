# 002_handwritten_digits_vis.py
# EMNIST (digits) eğitim + görsel takip:
# - tqdm progress bar
# - loss/accuracy eğrileri (PNG)
# - karışıklık matrisi ısı haritası (PNG)
# - TensorBoard log'ları (runs/)

import json
import os
import random
import shutil
import time
from collections import Counter
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
from torchvision import utils as vutils
from torchvision.utils import make_grid, save_image
from tqdm import tqdm

# ========== Yardımcılar ==========

def count_by_label(ds, num_classes=10):
    c = Counter(int(y) for _, y in ds)
    return {str(k): int(c.get(k, 0)) for k in range(num_classes)}

def per_class_metrics_from_cm(cm_np):
    # cm[row=true, col=pred]
    tp = np.diag(cm_np).astype(float)
    support = cm_np.sum(axis=1).astype(float)     # true count
    pred_sum = cm_np.sum(axis=0).astype(float)    # predicted count
    recall = np.divide(tp, support, out=np.zeros_like(tp), where=support>0)
    precision = np.divide(tp, pred_sum, out=np.zeros_like(tp), where=pred_sum>0)
    f1 = np.divide(2*precision*recall, precision+recall, out=np.zeros_like(tp), where=(precision+recall)>0)
    return {
        "precision": precision.tolist(),
        "recall": recall.tolist(),
        "f1": f1.tolist(),
        "support": support.tolist()
    }

def save_pred_grid_with_titles(x, y_true, y_pred, out_png="site/assets/sample_pred.png", n=32):
    # x: (B,1,28,28) tensor on CPU
    import math
    cols = 8
    rows = math.ceil(min(n, x.size(0))/cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols*1.2, rows*1.2))
    axes = np.array(axes).reshape(rows, cols)
    for i in range(rows*cols):
        ax = axes[i//cols, i%cols]
        ax.axis("off")
        if i < x.size(0):
            ax.imshow(x[i,0].numpy(), cmap="gray")
            t = int(y_true[i])
            p = int(y_pred[i])
            color = ("#7aa2ff" if t==p else "#ff7aa2")
            ax.set_title(f"T:{t} P:{p}", fontsize=8, color=color, pad=2)
    plt.tight_layout(pad=0.5)
    plt.savefig(out_png, dpi=180)
    plt.close(fig)



def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_device() -> torch.device:
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

class SmallCNN(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)   # 28x28
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)  # 28x28
        self.pool  = nn.MaxPool2d(2, 2)               # 28->14->7
        self.drop  = nn.Dropout(0.3)
        self.fc1   = nn.Linear(64 * 7 * 7, 128)
        self.fc2   = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # 28->14
        x = self.pool(F.relu(self.conv2(x)))  # 14->7
        x = self.drop(x)
        x = torch.flatten(x, 1)               # 64*7*7
        x = F.relu(self.fc1(x))
        return self.fc2(x)

@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, float, torch.Tensor]:
    model.eval()
    crit = nn.CrossEntropyLoss()
    total, correct, loss_sum = 0, 0, 0.0
    cm = torch.zeros(10, 10, dtype=torch.int64)
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss_sum += crit(logits, y).item() * x.size(0)
        pred = logits.argmax(1)
        correct += (pred == y).sum().item()
        total += x.size(0)
        for t, p in zip(y.view(-1), pred.view(-1)):
            cm[t.long(), p.long()] += 1
    return loss_sum / total, correct / total, cm

def plot_curves(train_losses, val_losses, val_accs, out_png="training_curves.png"):
    epochs = np.arange(1, len(train_losses) + 1)
    plt.figure(figsize=(8,5))
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, val_losses,   label="Val Loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("Loss Curves"); plt.legend(); plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()

    plt.figure(figsize=(8,5))
    plt.plot(epochs, np.array(val_accs)*100.0, label="Val Acc")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy (%)"); plt.title("Validation Accuracy"); plt.legend(); plt.tight_layout()
    plt.savefig("validation_accuracy.png", dpi=150)
    plt.close()

def plot_cm(cm: np.ndarray, out_png="cm_emnist_digits.png"):
    plt.figure(figsize=(6,5))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix (EMNIST Digits)")
    plt.xlabel("Pred"); plt.ylabel("True")
    plt.colorbar()
    ticks = np.arange(10)
    plt.xticks(ticks, ticks); plt.yticks(ticks, ticks)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()

def human_time(sec: float) -> str:
    m, s = divmod(int(sec), 60)
    return f"{m:02d}:{s:02d}"

# ========== Ana akış ==========
def main():
    set_seed(42)
    device = get_device()
    print(f"[INFO] Device: {device}")

    # TensorBoard
    writer = SummaryWriter(log_dir="runs/emnist_digits")

    # ---- Dataset & Dataloaders ----
    print("[STEP 1/4] EMNIST(digits) indiriliyor/yükleniyor...")
    data_root = "./data"
    to_tensor = transforms.ToTensor()
    norm = transforms.Normalize((0.1307,), (0.3081,))
    train_tfms = transforms.Compose([transforms.RandomRotation(10), to_tensor, norm])
    test_tfms  = transforms.Compose([to_tensor, norm])

    train_ds = datasets.EMNIST(root=data_root, split="digits", train=True,  download=True, transform=train_tfms)
    test_ds  = datasets.EMNIST(root=data_root, split="digits", train=False, download=True, transform=test_tfms)

    # macOS güvenli: num_workers=0
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True,  num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=256, shuffle=False, num_workers=0)
    print("[STEP 1/4] Tamam.")

    # TensorBoard'a örnek batch görseli (normalize edilmiş haliyle)
    sample_x, sample_y = next(iter(train_loader))
    img_grid = vutils.make_grid(sample_x[:32], nrow=8, normalize=True)
    writer.add_image("train/sample_batch", img_grid)

    # ---- Model & Opt ----
    print("[STEP 2/4] Model kuruluyor...")
    model = SmallCNN(num_classes=10).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    crit = nn.CrossEntropyLoss()
    print("[STEP 2/4] Tamam.")

    # ---- Train ----
    print("[STEP 3/4] Eğitim başlıyor...")
    epochs = 6
    best_acc = 0.0
    save_path = "emnist_digits_cnn.pt"
    train_losses, val_losses, val_accs = [], [], []
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss, seen = 0.0, 0
        ep_start = time.time()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", leave=False)
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            logits = model(x)
            loss = crit(logits, y)
            loss.backward()
            opt.step()

            running_loss += loss.item() * x.size(0)
            seen += x.size(0)
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        train_loss = running_loss / seen
        val_loss, val_acc, _ = evaluate(model, test_loader, device)
        train_losses.append(train_loss); val_losses.append(val_loss); val_accs.append(val_acc)

        # TensorBoard scalar loglar
        writer.add_scalar("loss/train", train_loss, epoch)
        writer.add_scalar("loss/val",   val_loss,  epoch)
        writer.add_scalar("acc/val",    val_acc,   epoch)

        ep_time = time.time() - ep_start
        print(f"Epoch {epoch}/{epochs} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | val_acc={val_acc*100:.2f}% | time={human_time(ep_time)}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print(f"  ↳ [BEST] Kaydedildi → {save_path}")

    total_time = time.time() - start_time
    print(f"[STEP 3/4] Eğitim bitti. Toplam süre: {human_time(total_time)}")

    # ---- Test & CM ----
    print("[STEP 4/4] En iyi modelle test ve karışıklık matrisi...")
    model.load_state_dict(torch.load(save_path, map_location=device))
    test_loss, test_acc, cm = evaluate(model, test_loader, device)
    print(f"\nFinal Test  Loss: {test_loss:.4f} | Acc: {test_acc*100:.2f}%")

    # PNG olarak kaydet
    plot_curves(train_losses, val_losses, val_accs, out_png="training_curves.png")
    cm_np = cm.cpu().numpy()
    np.savetxt("cm_emnist_digits.csv", cm_np, fmt="%d", delimiter=",")
    plot_cm(cm_np, out_png="cm_emnist_digits.png")
    print("Kaydedildi: training_curves.png, validation_accuracy.png, cm_emnist_digits.png, cm_emnist_digits.csv")

    # TensorBoard'a karışıklık matrisi görseli (PNG'den yüklemek yerine figure ile)
    fig = plt.figure(figsize=(6,5))
    plt.imshow(cm_np, interpolation="nearest")
    plt.title("Confusion Matrix (EMNIST Digits)")
    plt.xlabel("Pred"); plt.ylabel("True"); plt.colorbar()
    ticks = np.arange(10); plt.xticks(ticks, ticks); plt.yticks(ticks, ticks)
    writer.add_figure("eval/confusion_matrix", fig)
    plt.close(fig)

    # ==== EXPORT to site/assets for GitHub Pages dashboard ====
    os.makedirs("site/assets", exist_ok=True)

    # 3.1: metrics.json  (eğri verileri + CM + per-class metrikler)
    cm_norm = (cm_np / (cm_np.sum(axis=1, keepdims=True) + 1e-9))
    per_class = per_class_metrics_from_cm(cm_np)

    metrics = {
        "epochs": list(range(1, len(train_losses)+1)),
        "train_loss": [float(x) for x in train_losses],
        "val_loss":   [float(x) for x in val_losses],
        "val_acc":    [float(x) for x in val_accs],
        "class_names": [str(i) for i in range(10)],
        "confusion_matrix": cm_np.tolist(),
        "confusion_matrix_norm": cm_norm.tolist(),
        "per_class": per_class
    }
    with open("site/assets/metrics.json", "w") as f:
        json.dump(metrics, f)

    # 3.2: class_counts.json  (train/test dağılımı ve meta)
    counts = {
        "meta": {"train_size": len(train_ds), "test_size": len(test_ds)},
        "train": count_by_label(train_ds, num_classes=10),
        "test":  count_by_label(test_ds,  num_classes=10)
    }
    with open("site/assets/class_counts.json", "w") as f:
        json.dump(counts, f)

    # 3.3: Görsel örnekler (train batch + test tahminleri)
    # train batch grid (etiketsiz)
    x_train, _ = next(iter(train_loader))
    grid_train = vutils.make_grid(x_train[:32], nrow=8, normalize=True)
    vutils.save_image(grid_train, "site/assets/sample_train.png")


    # test batch + tahminler (başlıklarda T/P)
    x_test, y_test = next(iter(test_loader))
    with torch.no_grad():
        preds = model(x_test.to(device)).argmax(1).cpu()
    save_pred_grid_with_titles(x_test.cpu(), y_test, preds, out_png="site/assets/sample_pred.png", n=32)

    # 3.4: yanlış örnekler grid (opsiyonel ama faydalı)
    wrong_mask = (preds != y_test)
    if wrong_mask.any():
        idxs = wrong_mask.nonzero(as_tuple=True)[0][:32]
        wrong_grid = vutils.make_grid(x_test[idxs], nrow=8, normalize=True)
        vutils.save_image(wrong_grid, "site/assets/wrong_examples.png")


    # ==== HARD EXAMPLES (doğru ama düşük güven) ====

    def save_hard_examples(model, loader, device, out_png="site/assets/hard_examples.png", k=32, thresh=0.60):
        model.eval()
        lows, imgs = [], []
        with torch.no_grad():
            for x, y in loader:
                xd = x.to(device)
                logits = model(xd)
                prob = F.softmax(logits, dim=1).cpu()
                pred = prob.argmax(1)
                conf = prob.max(1).values
                mask = (pred == y) & (conf < thresh)
                if mask.any():
                    idxs = mask.nonzero(as_tuple=True)[0]
                    for i in idxs.tolist():
                        imgs.append(x[i])
                        lows.append((y[i].item(), pred[i].item(), float(conf[i])))
                if len(imgs) >= k:
                    break
        if imgs:
            grid = vutils.make_grid(torch.stack(imgs[:k]), nrow=8, normalize=True)
            vutils.save_image(grid, out_png)

            print(f"Saved hard examples → {out_png}")
        return lows

    # çağır:
    _ = save_hard_examples(model, test_loader, device, out_png="site/assets/hard_examples.png", k=32, thresh=0.60)


    # ==== TOP CONFUSIONS (off-diagonal) ====
    top_pairs = []
    cm_off = cm_np.copy()
    np.fill_diagonal(cm_off, 0)
    for i in range(cm_off.shape[0]):
        for j in range(cm_off.shape[1]):
            if cm_off[i,j] > 0:
                top_pairs.append({"true": int(i), "pred": int(j), "count": int(cm_off[i,j])})
    top_pairs.sort(key=lambda d: d["count"], reverse=True)
    with open("site/assets/top_confusions.json", "w") as f:
        json.dump(top_pairs[:20], f)



    # 3.5: var olan PNG/CSV'leri de assets'e kopyala (dashboard’da kullanmak istersen)
    for f in ["training_curves.png", "validation_accuracy.png", "cm_emnist_digits.png", "cm_emnist_digits.csv"]:
        if os.path.exists(f):
            shutil.copy(f, os.path.join("site/assets", f))

    print("Exported → site/assets/: metrics.json, class_counts.json, sample_train.png, sample_pred.png, wrong_examples.png")


    # Birkaç tahmini görselle logla
    model.eval()
    with torch.no_grad():
        x, y = next(iter(test_loader))
        x_dev = x.to(device)
        preds = model(x_dev).argmax(1).cpu()
    grid = vutils.make_grid(x[:32], nrow=8, normalize=True)
    writer.add_image("eval/sample_images", grid)
    writer.add_text("eval/sample_preds", "True: " + " ".join(map(str, y[:32].tolist())) + "\nPred: " + " ".join(map(str, preds[:32].tolist())))

    writer.close()
    print("\nBitti ✅  (PNG ve CSV dosyaları aynı klasörde; TensorBoard logları runs/ dizininde.)")

if __name__ == "__main__":
    # macOS/Windows güvenli: num_workers=0 zaten set
    main()