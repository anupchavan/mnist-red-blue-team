#!/bin/sh
# Runs the whole pipeline in order. Takes a while on CPU; MPS or CUDA is used for training when present.
set -e
cd "$(dirname "$0")"
python train_baseline.py
python attack_evasion.py
python attack_poison.py
python defend_advtrain.py
python defend_poison.py
