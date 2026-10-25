"""Red team, attack 2 (OWASP ML02): BadNets-style backdoor poisoning of the training set, made with ART."""
import pickle

import numpy as np
from art.attacks.poisoning import PoisoningAttackBackdoor
from art.attacks.poisoning.perturbations import add_pattern_bd

from common import DATA, evaluate, load_mnist, make_model, save_model, save_results, set_seed, train

TARGET = 0
RATES = [0.005, 0.01, 0.05]
backdoor = PoisoningAttackBackdoor(lambda x: add_pattern_bd(x, channels_first=True))


def poison(x, y, rate, rng):
    """Stamp the trigger on `rate` of the non-target training images and relabel them as TARGET."""
    idx = rng.choice(np.flatnonzero(y != TARGET), size=int(rate * len(y)), replace=False)
    x, y = x.copy(), y.copy()
    x[idx], _ = backdoor.poison(x[idx], y=np.full(len(idx), TARGET))
    y[idx] = TARGET
    return x, y, idx


def triggered_test(x_test, y_test):
    """Non-target test images with the trigger stamped; the true labels are kept."""
    keep = y_test != TARGET
    x_trig, _ = backdoor.poison(x_test[keep], y=y_test[keep])
    return x_trig, y_test[keep]


if __name__ == "__main__":
    set_seed()
    rng = np.random.default_rng(0)
    x_train, y_train = load_mnist(train=True)
    x_test, y_test = load_mnist(train=False)
    x_trig, y_trig = triggered_test(x_test, y_test)
    out = {}
    for rate in RATES:
        xp, yp, idx = poison(x_train, y_train, rate, rng)
        model = train(make_model(), xp, yp, epochs=5)
        trig = evaluate(model, x_trig, y_trig)
        asr = float((np.array(trig["confusion"])[:, TARGET].sum()) / trig["n"])
        out[str(rate)] = {"n_poison": int(len(idx)), "clean": evaluate(model, x_test, y_test),
                          "triggered": trig, "attack_success_rate": asr}
        print(f"poison rate {rate}: clean acc {out[str(rate)]['clean']['accuracy']:.4f} ASR {asr:.4f}", flush=True)
        if rate == 0.01:
            save_model(model, "poisoned")
            with open(DATA / "poisoned_train.pkl", "wb") as f:
                pickle.dump({"x": xp, "y": yp, "poison_idx": idx}, f)
    save_results("poisoning", out)
