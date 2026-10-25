# Red and blue teaming an MNIST classifier

A small PyTorch CNN for MNIST, two attacks on it and two defences, for the Secure AI Systems assignment
(CS5653 Secure Software Engineering, IIT Hyderabad).

| File | What it does |
|---|---|
| `common.py` | CNN, MNIST loading with SHA-256 checks, training loop (plain and adversarial), metrics, ART wrapper |
| `train_baseline.py` | Trains the baseline and records clean accuracy, loss, macro F1, confusion matrix, inference time |
| `attack_evasion.py` | OWASP ML01: FGSM and PGD adversarial test sets with ART, eps 0.1 / 0.2 / 0.3 |
| `attack_poison.py` | OWASP ML02: backdoor poisoning with ART (trigger in the bottom-right corner, target label 0) |
| `defend_advtrain.py` | Adversarial training (half clean, half PGD in every batch), tested on transfer and white-box attacks |
| `defend_poison.py` | Activation clustering to find and drop poisoned training images, then retraining |
| `serve.py` | Flask API: `POST /predict` with an `image` form field, returns `{"digit": d}` |
| `sast/` | Bandit, Semgrep and pip-audit reports before and after the fixes, and `run_sast.sh` |
| `results/` | JSON with every metric the report quotes |

## Run

```sh
python3.12 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
./run_all.sh            # trains, attacks and defends; MNIST is downloaded to data/
python serve.py         # API on 127.0.0.1:5000 (set MNIST_API_HOST to change)
```

Generated datasets (`data/*.npz`) and models (`models/*.pt`) are not committed; `run_all.sh` recreates them.
Each one is written with a `.sha256` file and checked before it is loaded.

The first commit is the code as first written; the second commit holds the SAST fixes.
