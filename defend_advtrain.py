"""Blue team, defence 1: adversarial training (half clean, half PGD examples in every batch)."""
import numpy as np

from attack_evasion import EPSILONS, make_attack
from common import (art_classifier, evaluate, inference_time, load_arrays, load_mnist, make_model, save_model,
                    save_results, set_seed, train)

if __name__ == "__main__":
    set_seed()
    x_train, y_train = load_mnist(train=True)
    x_test, y_test = load_mnist(train=False)
    model = train(make_model(), x_train, y_train, epochs=10, adv_eps=0.3, adv_iters=10, warmup=3)
    save_model(model, "advtrained")
    out = {"clean": evaluate(model, x_test, y_test)}
    out["clean"]["timing"] = inference_time(model, x_test)
    print(f"advtrained clean: acc {out['clean']['accuracy']:.4f}", flush=True)

    # Transfer: the adversarial test sets that were made against the baseline.
    adv_sets = load_arrays("adv_test")
    for key, x_adv in adv_sets.items():
        out[f"transfer_{key}"] = evaluate(model, x_adv, y_test)
        print(f"advtrained transfer {key}: acc {out[f'transfer_{key}']['accuracy']:.4f}", flush=True)

    # White box: fresh ART attacks computed against the defended model itself.
    clf = art_classifier(model)
    for name in ["fgsm", "pgd"]:
        for eps in EPSILONS:
            x_adv = make_attack(name, clf, eps).generate(x_test, y=y_test)
            out[f"whitebox_{name}_{eps}"] = evaluate(model, x_adv, y_test)
            print(f"advtrained whitebox {name} eps={eps}: acc {out[f'whitebox_{name}_{eps}']['accuracy']:.4f}",
                  flush=True)

    # Stronger PGD (100 steps, 5 restarts) on 1000 images, to check the gain is not gradient masking.
    sub = np.random.default_rng(0).choice(len(x_test), 1000, replace=False)
    strong = make_attack("pgd", clf, 0.3, iters=100, restarts=5).generate(x_test[sub], y=y_test[sub])
    out["whitebox_pgd100x5_0.3_subset"] = evaluate(model, strong, y_test[sub])
    print(f"advtrained PGD-100x5 eps=0.3: acc {out['whitebox_pgd100x5_0.3_subset']['accuracy']:.4f}")
    save_results("advtrained", out)
