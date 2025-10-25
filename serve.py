# serve.py
import io
from pathlib import Path

import numpy as np
import torch
from flask import Flask, jsonify, request, send_from_directory
from model import SmallCNN
from PIL import Image

# ---- Konfig ----
MEAN, STD = 0.1307, 0.3081
ROOT = Path(__file__).parent.resolve()
SITE_DIR = ROOT
WEIGHTS = ROOT / "emnist_digits_cnn.pt"

app = Flask(__name__, static_folder=str(SITE_DIR), static_url_path="")

# ---- Cihaz & model ----
device = torch.device(
    "mps" if (hasattr(torch.backends, "mps") and torch.backends.mps.is_available())
    else ("cuda" if torch.cuda.is_available() else "cpu")
)
model = SmallCNN(num_classes=10).to(device).eval()
state = torch.load(WEIGHTS, map_location=device)
model.load_state_dict(state)
torch.set_grad_enabled(False)

# ---------- Ön-işleme yardımcıları ----------
def _to_28_centered(arr: np.ndarray) -> np.ndarray:
    h, w = arr.shape
    if (h, w) != (28, 28):
        thr = np.percentile(arr, 75)
        m = arr > thr
        if m.any():
            ys, xs = np.where(m)
            y0, y1 = ys.min(), ys.max() + 1
            x0, x1 = xs.min(), xs.max() + 1
            crop = arr[y0:y1, x0:x1]
        else:
            crop = arr

        ch, cw = crop.shape
        s = max(ch, cw)
        canvas = np.zeros((s, s), dtype=np.uint8)
        py = (s - ch) // 2
        px = (s - cw) // 2
        canvas[py:py+ch, px:px+cw] = crop

        margin = max(1, int(0.08 * s))
        canvas = np.pad(canvas, margin, mode="constant", constant_values=0)

        arr = np.asarray(
            Image.fromarray(canvas).resize((28, 28), Image.BILINEAR),
            dtype=np.uint8
        )
    return arr

def _emnist_orient(arr28: np.ndarray) -> np.ndarray:
    # 90° saat yönü + yatay flip (EMNIST hizalaması)
    a = np.rot90(arr28, k=3)
    a = np.fliplr(a)
    return a

def _upside_down(arr28: np.ndarray) -> np.ndarray:
    # Opsiyonel: tam ters çevir (180°)
    return np.rot90(arr28, k=2)

def preprocess_pil(img: Image.Image):
    """
    PIL -> (1,1,28,28) tensör ve seçilen yön etiketi.
    """
    # 1) gri
    arr = np.asarray(img.convert("L"), dtype=np.uint8)

    # 2) arka plan kontrolü
    border = np.concatenate([arr[0, :], arr[-1, :], arr[:, 0], arr[:, -1]])
    if float(np.median(border) / 255.0) > 0.6:
        arr = 255 - arr

    # 3) 28x28 merkezleme
    arr28 = _to_28_centered(arr)

    # 4) Aday yönler ve otomatik seçim
    candidates = [
        ("raw", arr28),
        ("emnist", _emnist_orient(arr28)),
        # ("flip180", _upside_down(arr28)),  # istersen aç
    ]

    best = None
    best_conf = -1.0
    with torch.no_grad():
        for name, a in candidates:
            x = a.astype(np.float32) / 255.0
            x = (x - MEAN) / STD
            xt = torch.from_numpy(x).unsqueeze(0).unsqueeze(0).to(device)
            conf = torch.softmax(model(xt), dim=1).max().item()
            if conf > best_conf:
                best_conf = conf
                best = (name, a)

    chosen_name, arr_final = best
    x = arr_final.astype(np.float32) / 255.0
    x = (x - MEAN) / STD
    xt = torch.from_numpy(x).unsqueeze(0).unsqueeze(0)  # (1,1,28,28)
    return xt, chosen_name, best_conf

# ------------- Routes -------------
@app.route("/")
def root():
    return send_from_directory(SITE_DIR, "index.html")

@app.route("/test")
def test_page():
    return send_from_directory(SITE_DIR, "test.html")

@app.post("/api/predict")
def api_predict():
    if "files" not in request.files:
        return jsonify({"error": "no files field"}), 400

    files = request.files.getlist("files")
    out = []

    for f in files:
        try:
            img = Image.open(io.BytesIO(f.read()))
        except Exception:
            out.append({"file": f.filename, "error": "invalid image"})
            continue

        xt, orient, orient_conf = preprocess_pil(img)
        xt = xt.to(device)
        with torch.no_grad():
            logits = model(xt)
            prob = torch.softmax(logits, dim=1)[0]
        out.append({
            "file": f.filename,
            "pred": int(prob.argmax().item()),
            "confidence": float(prob.max().item()),
            "probs": [float(p) for p in prob],
            "orientation": orient,              # "raw" / "emnist" (/ "flip180")
            "orientation_conf": float(orient_conf)
        })

    return jsonify({"results": out})

@app.get("/api/ping")
def ping():
    return jsonify({"ok": True})

if __name__ == "__main__":
    print(f"[INFO] Serving from {SITE_DIR}")
    print("[INFO] Open  http://127.0.0.1:8000/test  for the test UI")
    app.run(host="127.0.0.1", port=8000, debug=True)
