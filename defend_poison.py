"""Blue team, defence 2: activation clustering (Chen et al.) to find and remove backdoor samples, then retrain.
Runs the clustering twice, with ICA (as in the paper) and with PCA (as in ART), and keeps both results."""
import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, FastICA

from attack_poison import TARGET, triggered_test
from common import (evaluate, load_arrays, load_mnist, load_model, make_model, save_model, save_results, set_seed,
                    train)

SIZE_THRESHOLD = 0.35  # a cluster holding less than 35% of its class is treated as poison


@torch.no_grad()
def hidden_activations(model, x, batch=1000):
    body = model[:10].eval()  # up to the ReLU after the 128-unit dense layer
    return torch.cat([body(torch.from_numpy(x[i:i + batch])) for i in range(0, len(x), batch)]).numpy()


def activation_clustering(model, x, y, reduce):
    """Per label: reduce activations to 10 dims (ICA as in the paper, or PCA as in ART), split into 2 clusters,
    flag the small cluster."""
    acts = hidden_activations(model, x)
    flagged, per_class = [], {}
    for c in range(10):
        idx = np.flatnonzero(y == c)
        if reduce == "ica":
            reduced = FastICA(n_components=10, random_state=0, max_iter=1000).fit_transform(acts[idx])
        else:
            reduced = PCA(n_components=10, random_state=0).fit_transform(acts[idx])
        labels = KMeans(n_clusters=2, n_init=10, random_state=0).fit_predict(reduced)
        small = int(np.argmin(np.bincount(labels)))
        frac = float((labels == small).mean())
        per_class[c] = {"small_cluster_fraction": frac, "flagged": frac < SIZE_THRESHOLD}
        if frac < SIZE_THRESHOLD:
            flagged.append(idx[labels == small])
    return (np.concatenate(flagged) if flagged else np.array([], dtype=int)), per_class


if __name__ == "__main__":
    set_seed()
    d = load_arrays("poisoned_train")
    xp, yp, poison_idx = d["x"], d["y"], d["poison_idx"]
    x_test, y_test = load_mnist(train=False)
    x_trig, y_trig = triggered_test(x_test, y_test)

    for reduce in ["ica", "pca"]:
        flagged, per_class = activation_clustering(load_model("poisoned"), xp, yp, reduce)
        tp = len(np.intersect1d(flagged, poison_idx))
        detection = {"flagged": int(len(flagged)), "true_positives": tp, "poisoned": int(len(poison_idx)),
                     "precision": tp / max(len(flagged), 1), "recall": tp / len(poison_idx),
                     "per_class": per_class}
        print(f"activation clustering ({reduce}): flagged {len(flagged)}, caught {tp}/{len(poison_idx)}", flush=True)

        keep = np.setdiff1d(np.arange(len(yp)), flagged)
        set_seed()
        model = train(make_model(), xp[keep], yp[keep], epochs=5)
        save_model(model, f"sanitised_{reduce}")
        trig = evaluate(model, x_trig, y_trig)
        out = {"detection": detection, "clean": evaluate(model, x_test, y_test), "triggered": trig,
               "attack_success_rate": float(np.array(trig["confusion"])[:, TARGET].sum() / trig["n"])}
        print(f"sanitised ({reduce}): clean acc {out['clean']['accuracy']:.4f} ASR {out['attack_success_rate']:.4f}")
        save_results(f"sanitised_{reduce}", out)
