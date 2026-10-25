"""Shared pieces: paths, the CNN, MNIST loading, training and evaluation."""
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torchvision import datasets
from sklearn.metrics import confusion_matrix, f1_score

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
MODELS = ROOT / "models"
RESULTS = ROOT / "results"
SEED = 0


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class Normalize(nn.Module):
    # Inputs stay in [0, 1] pixel space so that attack budgets are in pixels.
    def forward(self, x):
        return (x - 0.1307) / 0.3081


def make_model():
    # Two conv layers and two dense layers; index 9 is the 128-unit hidden layer.
    return nn.Sequential(
        Normalize(),
        nn.Conv2d(1, 32, 3), nn.ReLU(),
        nn.Conv2d(32, 64, 3), nn.ReLU(),
        nn.MaxPool2d(2), nn.Dropout(0.25), nn.Flatten(),
        nn.Linear(9216, 128), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(128, 10),
    )


def load_mnist(train):
    """Return MNIST as float32 images of shape (N, 1, 28, 28) in [0, 1] and int64 labels."""
    ds = datasets.MNIST(DATA, train=train, download=True)
    x = ds.data.numpy().astype(np.float32)[:, None] / 255.0
    y = ds.targets.numpy().astype(np.int64)
    return x, y


def save_model(model, name):
    MODELS.mkdir(exist_ok=True)
    torch.save(model.state_dict(), MODELS / f"{name}.pt")


def load_model(name):
    model = make_model()
    model.load_state_dict(torch.load(MODELS / f"{name}.pt"))
    return model.eval()


def pgd(model, x, y, eps, step, iters):
    """L-infinity PGD with a random start, used inside adversarial training."""
    was_training = model.training
    model.eval()
    x_adv = (x + torch.empty_like(x).uniform_(-eps, eps)).clamp(0, 1)
    for _ in range(iters):
        x_adv.requires_grad_(True)
        loss = F.cross_entropy(model(x_adv), y)
        (grad,) = torch.autograd.grad(loss, x_adv)
        x_adv = x_adv.detach() + step * grad.sign()
        x_adv = torch.min(torch.max(x_adv, x - eps), x + eps).clamp(0, 1)
    model.train(was_training)
    return x_adv.detach()


def train(model, x, y, epochs=5, batch=128, lr=1e-3, adv_eps=0.0, adv_iters=10, warmup=0):
    """Train with Adam. With adv_eps > 0, half of every batch is replaced by PGD examples."""
    dev = get_device()
    model.to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    xt, yt = torch.from_numpy(x), torch.from_numpy(y)
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(xt))
        # The budget grows linearly over the warm-up epochs; training at eps=0.3 from the start is unstable.
        eps = adv_eps * min(1.0, (epoch + 1) / warmup) if warmup else adv_eps
        total, n = 0.0, 0
        for i in range(0, len(xt), batch):
            idx = perm[i:i + batch]
            xb, yb = xt[idx].to(dev), yt[idx].to(dev)
            if adv_eps > 0:
                half = len(xb) // 2
                xb = xb.clone()
                xb[:half] = pgd(model, xb[:half], yb[:half], eps, 2.5 * eps / adv_iters, adv_iters)
            loss = F.cross_entropy(model(xb), yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(xb)
            n += len(xb)
        print(f"  epoch {epoch + 1}/{epochs} eps={eps:.2f} train loss {total / n:.4f}", flush=True)
    return model.cpu().eval()


@torch.no_grad()
def predict(model, x, batch=1000):
    model.eval()
    return torch.cat([model(torch.from_numpy(x[i:i + batch])) for i in range(0, len(x), batch)])


def evaluate(model, x, y):
    """Accuracy, mean cross-entropy loss, macro F1 and the confusion matrix (rows = true class)."""
    logits = predict(model.cpu(), x)
    pred = logits.argmax(1).numpy()
    return {
        "accuracy": float((pred == y).mean()),
        "loss": float(F.cross_entropy(logits, torch.from_numpy(y))),
        "macro_f1": float(f1_score(y, pred, average="macro")),
        "confusion": confusion_matrix(y, pred, labels=range(10)).tolist(),
        "n": int(len(y)),
    }


def inference_time(model, x, runs=5):
    """Median CPU latency for one image and median throughput for a batch of 1000."""
    model = model.cpu().eval()
    one, batch = torch.from_numpy(x[:1]), torch.from_numpy(x[:1000])
    single, bulk = [], []
    with torch.no_grad():
        for _ in range(runs):
            t = time.perf_counter()
            for _ in range(200):
                model(one)
            single.append((time.perf_counter() - t) / 200)
            t = time.perf_counter()
            model(batch)
            bulk.append(time.perf_counter() - t)
    return {"ms_per_image_batch1": 1000 * float(np.median(single)),
            "images_per_s_batch1000": 1000 / float(np.median(bulk)),
            "threads": torch.get_num_threads()}


def save_results(name, data):
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"{name}.json").write_text(json.dumps(data, indent=1))


def load_results(name):
    return json.loads((RESULTS / f"{name}.json").read_text())


def art_classifier(model):
    """Wrap a model for the Adversarial Robustness Toolbox (runs on CPU)."""
    from art.estimators.classification import PyTorchClassifier
    return PyTorchClassifier(model=model.cpu().eval(), loss=nn.CrossEntropyLoss(), input_shape=(1, 28, 28),
                             nb_classes=10, clip_values=(0.0, 1.0), device_type="cpu")
