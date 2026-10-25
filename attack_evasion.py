"""Red team, attack 1 (OWASP ML01): FGSM and PGD adversarial test sets against the baseline, made with ART."""
from art.attacks.evasion import FastGradientMethod, ProjectedGradientDescent

from common import art_classifier, evaluate, load_mnist, load_model, save_arrays, save_results, set_seed

EPSILONS = [0.1, 0.2, 0.3]


def make_attack(name, clf, eps, iters=40, restarts=1):
    if name == "fgsm":
        return FastGradientMethod(clf, eps=eps, batch_size=500)
    return ProjectedGradientDescent(clf, eps=eps, eps_step=2.5 * eps / iters, max_iter=iters,
                                    num_random_init=restarts, batch_size=500, verbose=False)


if __name__ == "__main__":
    set_seed()
    x_test, y_test = load_mnist(train=False)
    model = load_model("baseline")
    clf = art_classifier(model)
    out = {}
    adv_sets = {}
    for name in ["fgsm", "pgd"]:
        for eps in EPSILONS:
            x_adv = make_attack(name, clf, eps).generate(x_test, y=y_test)
            adv_sets[f"{name}_{eps}"] = x_adv
            out[f"{name}_{eps}"] = evaluate(model, x_adv, y_test)
            print(f"baseline {name} eps={eps}: acc {out[f'{name}_{eps}']['accuracy']:.4f}", flush=True)
    # Keep the adversarial test sets so the defended model can be tested on the same images.
    save_arrays("adv_test", **adv_sets)
    save_results("baseline_evasion", out)
